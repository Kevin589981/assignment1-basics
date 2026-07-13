use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Pair(Vec<u8>, Vec<u8>);

type Word = Vec<Vec<u8>>;

fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

fn is_space(ch: char) -> bool {
    ch.is_whitespace()
}

fn next_char(text: &str, idx: usize) -> Option<(char, usize)> {
    text[idx..].chars().next().map(|ch| (ch, ch.len_utf8()))
}

fn starts_with_contraction(text: &str, idx: usize) -> Option<usize> {
    let rest = &text[idx..];
    for suffix in ["'ll", "'ve", "'re", "'s", "'d", "'m", "'t"] {
        if rest.starts_with(suffix) {
            return Some(suffix.len());
        }
    }
    None
}

fn consume_run<F>(text: &str, mut idx: usize, pred: F) -> usize
where
    F: Fn(char) -> bool,
{
    while idx < text.len() {
        let Some((ch, width)) = next_char(text, idx) else {
            break;
        };
        if !pred(ch) {
            break;
        }
        idx += width;
    }
    idx
}

fn consume_whitespace_like_gpt2(text: &str, start: usize) -> (usize, usize) {
    let mut idx = start;
    let mut last_start = start;
    let mut count = 0usize;
    while idx < text.len() {
        let Some((ch, width)) = next_char(text, idx) else {
            break;
        };
        if !is_space(ch) {
            break;
        }
        last_start = idx;
        idx += width;
        count += 1;
    }
    if idx < text.len() && count > 1 {
        (last_start, last_start)
    } else {
        (idx, idx)
    }
}

fn pretokenize_segment(text: &str, counts: &mut HashMap<Word, u64>) {
    let mut idx = 0;
    while idx < text.len() {
        if let Some(width) = starts_with_contraction(text, idx) {
            add_pretoken(&text.as_bytes()[idx..idx + width], counts);
            idx += width;
            continue;
        }

        let start = idx;
        let Some((ch, width)) = next_char(text, idx) else {
            break;
        };
        if ch == ' ' {
            if let Some((next, next_width)) = next_char(text, idx + width) {
                if is_letter(next) {
                    idx = consume_run(text, idx + width + next_width, is_letter);
                    add_pretoken(&text.as_bytes()[start..idx], counts);
                    continue;
                }
                if is_number(next) {
                    idx = consume_run(text, idx + width + next_width, is_number);
                    add_pretoken(&text.as_bytes()[start..idx], counts);
                    continue;
                }
                if !is_space(next) && !is_letter(next) && !is_number(next) {
                    idx = consume_run(text, idx + width + next_width, |c| {
                        !is_space(c) && !is_letter(c) && !is_number(c)
                    });
                    add_pretoken(&text.as_bytes()[start..idx], counts);
                    continue;
                }
            }
        }

        if is_letter(ch) {
            idx = consume_run(text, idx + width, is_letter);
        } else if is_number(ch) {
            idx = consume_run(text, idx + width, is_number);
        } else if !is_space(ch) {
            idx = consume_run(text, idx + width, |c| {
                !is_space(c) && !is_letter(c) && !is_number(c)
            });
        } else {
            let (match_end, next_idx) = consume_whitespace_like_gpt2(text, idx);
            add_pretoken(&text.as_bytes()[start..match_end], counts);
            idx = next_idx;
            continue;
        }
        add_pretoken(&text.as_bytes()[start..idx], counts);
    }
}

fn add_pretoken(bytes: &[u8], counts: &mut HashMap<Word, u64>) {
    if bytes.is_empty() {
        return;
    }
    let word: Word = bytes.iter().map(|b| vec![*b]).collect();
    *counts.entry(word).or_insert(0) += 1;
}

fn split_specials<'a>(text: &'a str, special_tokens: &[String]) -> Vec<&'a str> {
    if special_tokens.is_empty() {
        return vec![text];
    }
    let mut segments = Vec::new();
    let mut idx = 0;
    while idx < text.len() {
        let mut best: Option<(&str, usize)> = None;
        for token in special_tokens {
            if text[idx..].starts_with(token) {
                let len = token.len();
                if best.map_or(true, |(_, old_len)| len > old_len) {
                    best = Some((token.as_str(), len));
                }
            }
        }
        if let Some((_, len)) = best {
            idx += len;
            continue;
        }
        let start = idx;
        while idx < text.len() {
            let mut found = false;
            for token in special_tokens {
                if text[idx..].starts_with(token) {
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
            let Some((_, width)) = next_char(text, idx) else {
                break;
            };
            idx += width;
        }
        if start < idx {
            segments.push(&text[start..idx]);
        }
    }
    segments
}

fn build_indexes(
    word_counts: &HashMap<Word, u64>,
) -> (HashMap<Pair, u64>, HashMap<Pair, HashSet<Word>>) {
    let mut pair_counts: HashMap<Pair, u64> = HashMap::new();
    let mut pair_to_words: HashMap<Pair, HashSet<Word>> = HashMap::new();
    for (word, count) in word_counts {
        for pair in word.windows(2) {
            let p = Pair(pair[0].clone(), pair[1].clone());
            *pair_counts.entry(p.clone()).or_insert(0) += *count;
            pair_to_words.entry(p).or_default().insert(word.clone());
        }
    }
    (pair_counts, pair_to_words)
}

fn merge_word(word: &[Vec<u8>], pair: &Pair) -> Word {
    let mut merged = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
            let mut token = word[i].clone();
            token.extend_from_slice(&word[i + 1]);
            merged.push(token);
            i += 2;
        } else {
            merged.push(word[i].clone());
            i += 1;
        }
    }
    merged
}

fn dec_pair_count(pair_counts: &mut HashMap<Pair, u64>, pair: &Pair, amount: u64) {
    if let Some(value) = pair_counts.get_mut(pair) {
        if *value > amount {
            *value -= amount;
        } else {
            pair_counts.remove(pair);
        }
    }
}

fn best_pair(pair_counts: &HashMap<Pair, u64>) -> Option<Pair> {
    let mut best: Option<(&Pair, u64)> = None;
    for (pair, count) in pair_counts {
        if best.map_or(true, |(old_pair, old_count)| {
            *count > old_count || (*count == old_count && pair_gt(pair, old_pair))
        }) {
            best = Some((pair, *count));
        }
    }
    best.map(|(pair, _)| pair.clone())
}

fn pair_gt(a: &Pair, b: &Pair) -> bool {
    match a.0.cmp(&b.0) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => a.1 > b.1,
    }
}

fn hex(bytes: &[u8]) -> String {
    const CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(CHARS[(b >> 4) as usize] as char);
        out.push(CHARS[(b & 0x0f) as usize] as char);
    }
    out
}

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_json(
    path: &str,
    vocab: &[Vec<u8>],
    merges: &[Pair],
    special_tokens: &[String],
) -> io::Result<()> {
    let mut out = String::new();
    out.push_str("{\"vocab\":[");
    for (idx, token) in vocab.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&format!("[{},\"{}\"]", idx, hex(token)));
    }
    out.push_str("],\"merges\":[");
    for (idx, pair) in merges.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&format!("[\"{}\",\"{}\"]", hex(&pair.0), hex(&pair.1)));
    }
    out.push_str("],\"special_tokens\":[");
    for (idx, token) in special_tokens.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(token));
        out.push('"');
    }
    out.push_str("]}");
    fs::write(path, out)
}

fn parse_args() -> Result<(String, String, usize, Vec<String>, usize, Option<String>), String> {
    let mut input = None;
    let mut output = None;
    let mut vocab_size = None;
    let mut special_tokens = Vec::new();
    let mut progress_interval = 500usize;
    let mut dump_pretokens = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next(),
            "--output-json" => output = args.next(),
            "--vocab-size" => {
                vocab_size = Some(args.next().ok_or("missing --vocab-size value")?.parse().map_err(|_| "bad vocab size")?)
            }
            "--special-token" => special_tokens.push(args.next().ok_or("missing --special-token value")?),
            "--progress-interval" => {
                progress_interval = args
                    .next()
                    .ok_or("missing --progress-interval value")?
                    .parse()
                    .map_err(|_| "bad progress interval")?;
            }
            "--dump-pretokens" => dump_pretokens = args.next(),
            _ => return Err(format!("unknown argument: {}", arg)),
        }
    }
    Ok((
        input.ok_or("missing --input")?,
        output.ok_or("missing --output-json")?,
        vocab_size.ok_or("missing --vocab-size")?,
        special_tokens,
        progress_interval,
        dump_pretokens,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (input, output, vocab_size, special_tokens, progress_interval, dump_pretokens) =
        parse_args().map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let start = Instant::now();
    let text = fs::read_to_string(&input)?.replace("\r\n", "\n").replace('\r', "\n");
    let mut word_counts: HashMap<Word, u64> = HashMap::new();
    for segment in split_specials(&text, &special_tokens) {
        pretokenize_segment(segment, &mut word_counts);
    }
    eprintln!(
        "pretokenized unique_words={} elapsed={:.1}s",
        word_counts.len(),
        start.elapsed().as_secs_f64()
    );
    if let Some(path) = dump_pretokens {
        let mut lines: Vec<String> = word_counts
            .iter()
            .map(|(word, count)| {
                let mut bytes = Vec::new();
                for token in word {
                    bytes.extend_from_slice(token);
                }
                format!("{}\t{}", hex(&bytes), count)
            })
            .collect();
        lines.sort();
        fs::write(path, lines.join("\n"))?;
    }

    let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
    for token in &special_tokens {
        let bytes = token.as_bytes().to_vec();
        if !vocab.iter().any(|item| item == &bytes) {
            vocab.push(bytes);
        }
    }

    let (mut pair_counts, mut pair_to_words) = build_indexes(&word_counts);
    let mut merges: Vec<Pair> = Vec::new();
    while vocab.len() < vocab_size {
        let Some(pair) = best_pair(&pair_counts) else {
            break;
        };
        let mut merged_token = pair.0.clone();
        merged_token.extend_from_slice(&pair.1);
        vocab.push(merged_token);
        merges.push(pair.clone());

        let affected: Vec<Word> = pair_to_words
            .remove(&pair)
            .unwrap_or_default()
            .into_iter()
            .collect();
        for word in affected {
            let Some(count) = word_counts.remove(&word) else {
                continue;
            };
            let new_word = merge_word(&word, &pair);
            if new_word == word {
                *word_counts.entry(word).or_insert(0) += count;
                continue;
            }
            for old in word.windows(2) {
                let old_pair = Pair(old[0].clone(), old[1].clone());
                dec_pair_count(&mut pair_counts, &old_pair, count);
                if old_pair != pair {
                    if let Some(words) = pair_to_words.get_mut(&old_pair) {
                        words.remove(&word);
                        if words.is_empty() {
                            pair_to_words.remove(&old_pair);
                        }
                    }
                }
            }
            *word_counts.entry(new_word.clone()).or_insert(0) += count;
            for new in new_word.windows(2) {
                let new_pair = Pair(new[0].clone(), new[1].clone());
                *pair_counts.entry(new_pair.clone()).or_insert(0) += count;
                pair_to_words.entry(new_pair).or_default().insert(new_word.clone());
            }
        }

        if progress_interval > 0 && merges.len() % progress_interval == 0 {
            eprintln!(
                "merge {}/{} pair_counts={} elapsed={:.1}s",
                merges.len(),
                vocab_size.saturating_sub(256 + special_tokens.len()),
                pair_counts.len(),
                start.elapsed().as_secs_f64()
            );
            io::stderr().flush().ok();
        }
    }
    write_json(&output, &vocab, &merges, &special_tokens)?;
    eprintln!(
        "done vocab={} merges={} elapsed={:.1}s",
        vocab.len(),
        merges.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
