from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import os

import regex as re


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _special_pattern(special_tokens: list[str] | None) -> re.Pattern | None:
    if not special_tokens:
        return None
    ordered = sorted(special_tokens, key=len, reverse=True)
    return re.compile("(" + "|".join(re.escape(token) for token in ordered) + ")")


def _pretokenize(text: str) -> list[bytes]:
    return [match.group(0).encode("utf-8") for match in PAT.finditer(text)]


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_pattern = _special_pattern(self.special_tokens)
        self.special_to_id = {token: self.byte_to_id[token.encode("utf-8")] for token in self.special_tokens}

    def _encode_bytes(self, word: bytes) -> list[int]:
        tokens = [bytes([b]) for b in word]
        if len(tokens) < 2:
            return [self.byte_to_id[token] for token in tokens]
        while True:
            best_rank = None
            best_idx = None
            for i in range(len(tokens) - 1):
                rank = self.merge_ranks.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx is None:
                break
            tokens[best_idx : best_idx + 2] = [tokens[best_idx] + tokens[best_idx + 1]]
        return [self.byte_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        parts = self.special_pattern.split(text) if self.special_pattern is not None else [text]
        for part in parts:
            if part == "":
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                for word in _pretokenize(part):
                    ids.extend(self._encode_bytes(word))
        return ids

    def encode_iterable(self, iterable: Iterable[str]):
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[int(idx)] for idx in ids).decode("utf-8", errors="replace")


def _split_specials(text: str, special_tokens: list[str]) -> list[str]:
    pattern = _special_pattern(special_tokens)
    if pattern is None:
        return [text]
    return [part for part in pattern.split(text) if part and part not in special_tokens]


def _word_pair_counts(word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, count in word_counts.items():
        for pair in zip(word, word[1:]):
            counts[pair] += count
    return counts


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for segment in _split_specials(text, special_tokens):
        for pretoken in _pretokenize(segment):
            word_counts[tuple(bytes([b]) for b in pretoken)] += 1

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pair_counts = _word_pair_counts(word_counts)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        next_counts: Counter[tuple[bytes, ...]] = Counter()
        for word, count in word_counts.items():
            next_counts[_merge_word(word, best_pair)] += count
        word_counts = next_counts
    return vocab, merges
