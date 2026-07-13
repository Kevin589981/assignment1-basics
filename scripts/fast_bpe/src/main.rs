use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Pair(Vec<u8>, Vec<u8>);

type Word = Vec<Vec<u8>>;

#[derive(Debug)]
struct MergeUpdate {
    old_word: Word,
    new_word: Word,
    count: u64,
    old_pairs: Vec<Pair>,
    new_pairs: Vec<Pair>,
}

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

fn merge_word_counts(
    mut left: HashMap<Word, u64>,
    right: HashMap<Word, u64>,
) -> HashMap<Word, u64> {
    for (word, count) in right {
        *left.entry(word).or_insert(0) += count;
    }
    left
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
    word_counts
        .par_iter()
        .fold(
            || {
                (
                    HashMap::<Pair, u64>::new(),
                    HashMap::<Pair, HashSet<Word>>::new(),
                )
            },
            |(mut pair_counts, mut pair_to_words), (word, count)| {
                for pair in word.windows(2) {
                    let p = Pair(pair[0].clone(), pair[1].clone());
                    *pair_counts.entry(p.clone()).or_insert(0) += *count;
                    pair_to_words.entry(p).or_default().insert(word.clone());
                }
                (pair_counts, pair_to_words)
            },
        )
        .reduce(
            || {
                (
                    HashMap::<Pair, u64>::new(),
                    HashMap::<Pair, HashSet<Word>>::new(),
                )
            },
            |(mut left_counts, mut left_words), (right_counts, right_words)| {
                for (pair, count) in right_counts {
                    *left_counts.entry(pair).or_insert(0) += count;
                }
                for (pair, words) in right_words {
                    left_words.entry(pair).or_default().extend(words);
                }
                (left_counts, left_words)
            },
        )
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
    pair_counts
        .par_iter()
        .max_by(|(left_pair, left_count), (right_pair, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| cmp_pair(left_pair, right_pair))
        })
        .map(|(pair, _)| pair.clone())
}

fn cmp_pair(a: &Pair, b: &Pair) -> Ordering {
    match a.0.cmp(&b.0) {
        Ordering::Equal => a.1.cmp(&b.1),
        order => order,
    }
}

fn merge_update(word: Word, count: u64, pair: &Pair) -> MergeUpdate {
    let old_pairs = word
        .windows(2)
        .map(|old| Pair(old[0].clone(), old[1].clone()))
        .collect();
    let new_word = merge_word(&word, pair);
    let new_pairs = new_word
        .windows(2)
        .map(|new| Pair(new[0].clone(), new[1].clone()))
        .collect();
    MergeUpdate {
        old_word: word,
        new_word,
        count,
        old_pairs,
        new_pairs,
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

fn parse_args() -> Result<
    (
        String,
        String,
        usize,
        Vec<String>,
        usize,
        usize,
        Option<String>,
    ),
    String,
> {
    let mut input = None;
    let mut output = None;
    let mut vocab_size = None;
    let mut special_tokens = Vec::new();
    let mut progress_interval = 500usize;
    let mut num_workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut dump_pretokens = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next(),
            "--output-json" => output = args.next(),
            "--vocab-size" => {
                vocab_size = Some(
                    args.next()
                        .ok_or("missing --vocab-size value")?
                        .parse()
                        .map_err(|_| "bad vocab size")?,
                )
            }
            "--special-token" => {
                special_tokens.push(args.next().ok_or("missing --special-token value")?)
            }
            "--progress-interval" => {
                progress_interval = args
                    .next()
                    .ok_or("missing --progress-interval value")?
                    .parse()
                    .map_err(|_| "bad progress interval")?;
            }
            "--num-workers" => {
                num_workers = args
                    .next()
                    .ok_or("missing --num-workers value")?
                    .parse()
                    .map_err(|_| "bad num workers")?;
                if num_workers == 0 {
                    return Err("--num-workers must be positive".to_string());
                }
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
        num_workers,
        dump_pretokens,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (input, output, vocab_size, special_tokens, progress_interval, num_workers, dump_pretokens) =
        parse_args().map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build_global()
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
    let start = Instant::now();
    let text = fs::read_to_string(&input)?
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let segments = split_specials(&text, &special_tokens);
    let mut word_counts: HashMap<Word, u64> = segments
        .par_iter()
        .map(|segment| {
            let mut counts = HashMap::new();
            pretokenize_segment(segment, &mut counts);
            counts
        })
        .reduce(HashMap::new, merge_word_counts);
    eprintln!(
        "pretokenized unique_words={} segments={} workers={} elapsed={:.1}s",
        word_counts.len(),
        segments.len(),
        rayon::current_num_threads(),
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

    let index_start = Instant::now();
    let (mut pair_counts, mut pair_to_words) = build_indexes(&word_counts);
    eprintln!(
        "indexed pair_counts={} workers={} elapsed={:.1}s total_elapsed={:.1}s",
        pair_counts.len(),
        rayon::current_num_threads(),
        index_start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64()
    );
    let mut merges: Vec<Pair> = Vec::new();
    while vocab.len() < vocab_size {
        let merge_start = Instant::now();
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
        let mut affected_with_counts = Vec::with_capacity(affected.len());
        for word in affected {
            if let Some(count) = word_counts.remove(&word) {
                affected_with_counts.push((word, count));
            }
        }

        let affected_count = affected_with_counts.len();
        let updates: Vec<MergeUpdate> = affected_with_counts
            .into_par_iter()
            .map(|(word, count)| merge_update(word, count, &pair))
            .collect();

        for update in updates {
            if update.new_word == update.old_word {
                *word_counts.entry(update.old_word).or_insert(0) += update.count;
                continue;
            }
            for old_pair in update.old_pairs {
                dec_pair_count(&mut pair_counts, &old_pair, update.count);
                if old_pair != pair {
                    if let Some(words) = pair_to_words.get_mut(&old_pair) {
                        words.remove(&update.old_word);
                        if words.is_empty() {
                            pair_to_words.remove(&old_pair);
                        }
                    }
                }
            }
            *word_counts.entry(update.new_word.clone()).or_insert(0) += update.count;
            for new_pair in update.new_pairs {
                *pair_counts.entry(new_pair.clone()).or_insert(0) += update.count;
                pair_to_words
                    .entry(new_pair)
                    .or_default()
                    .insert(update.new_word.clone());
            }
        }

        if progress_interval > 0 && merges.len() % progress_interval == 0 {
            eprintln!(
                "merge {}/{} pair_counts={} affected_words={} merge_elapsed={:.2}s workers={} elapsed={:.1}s",
                merges.len(),
                vocab_size.saturating_sub(256 + special_tokens.len()),
                pair_counts.len(),
                affected_count,
                merge_start.elapsed().as_secs_f64(),
                rayon::current_num_threads(),
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
