"""Stage-1 keyword matcher: pypinyin-based keyword lexicon.

Scans ASR word streams (with absolute timestamps) for trigger keywords,
using pypinyin syllable-aligned exact matching only (no fuzzy matching).
In DEMO mode also emits any-voice triggers on VAD-positive chunks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pypinyin import lazy_pinyin

logger = logging.getLogger("keyword_matcher")

KEYWORD_LIBRARY: dict[str, list[str]] = {
    "买菜交易": [
        "多少钱", "付钱", "土豆", "白菜",
        "现金", "红菜苔", "快递",
    ],
}


@dataclass
class Hit:
    keyword: str
    violation_type: str
    abs_timestamp: float
    sentence_text: str
    match_mode: str
    source: str = "keyword"


@dataclass
class Word:
    text: str
    abs_begin: float
    abs_end: float


def _pinyin_syllables(text: str) -> list[str]:
    """Return per-character pinyin syllables, e.g. '快递' -> ['kuai','di']."""
    return list(lazy_pinyin(text))


def _word_syllables(words: list[Word]):
    """Flatten all words into a syllable list + syllable->word index map."""
    syllables: list[str] = []
    syl_to_word: list[int] = []
    for wi, w in enumerate(words):
        for s in _pinyin_syllables(w.text):
            syllables.append(s)
            syl_to_word.append(wi)
    return syllables, syl_to_word


def scan_transcript(words, demo_mode=False):
    """Scan a chunk word stream for keyword hits (exact pinyin match only)."""
    if not words:
        return []
    syllables, syl_to_word = _word_syllables(words)
    if not syllables:
        return []
    full_text = "".join(w.text for w in words)
    n_syl = len(syllables)
    hits: list[Hit] = []
    seen: set[tuple[str, int]] = set()

    for vtype, kws in KEYWORD_LIBRARY.items():
        for kw in kws:
            kw_syls = _pinyin_syllables(kw)
            if not kw_syls:
                continue
            kw_n = len(kw_syls)
            for start in range(n_syl - kw_n + 1):
                window_syls = syllables[start:start + kw_n]
                if window_syls != kw_syls:
                    continue
                wi = syl_to_word[start]
                key = (kw, wi)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(Hit(
                    keyword=kw, violation_type=vtype,
                    abs_timestamp=words[wi].abs_begin,
                    sentence_text=full_text,
                    match_mode="exact",
                ))

    if not hits and demo_mode:
        w0 = words[0]
        hits.append(Hit(
            keyword="(人声触发)", violation_type="演示触发",
            abs_timestamp=w0.abs_begin,
            sentence_text=full_text,
            match_mode="exact", source="demo_voice",
        ))
    hits.sort(key=lambda h: h.abs_timestamp)
    return hits


def build_context(words, hit_idx_word, span=5):
    lo = max(0, hit_idx_word - span)
    hi = min(len(words), hit_idx_word + span + 1)
    return "".join(w.text for w in words[lo:hi])
