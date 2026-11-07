from __future__ import annotations

import math
import logging
import re
from collections import Counter
from pathlib import Path

import jieba

jieba.setLogLevel(logging.ERROR)

_TMP = Path(__file__).resolve().parents[1] / "runtime" / "tmp"
if _TMP.exists():
    jieba.dt.tmp_dir = str(_TMP)
    jieba.dt.cache_file = "jieba.cache"

STOPWORDS = set(
    """
的 了 是 在 和 与 及 或 而 但 就 都 也 又 很 非常 我们 你们 他们 它 这 那 这个 那个
一个 一种 可以 能够 通过 进行 以及 因为 所以 如果 那么 然后 就是 这样 那样 什么
怎么 为什么 如何 已经 还是 但是 并且 同时 例如 比如 一些 这些 那些 自己 大家 时候
今天 下面 上面 首先 其次 最后 其实 比较 主要 相关 内容 问题 方法 情况 方面 过程
许多 这种 这时 那时 其中 之后 之前 同时 于是 仍然 只是 一个 一种
the a an and or but to of in on for with from by is are was were be been being this that
these those it its they them we you he she i as at into about over under can could would
should may might will just also more most much many some any not no if then than so because
sure really okay right now well thing things get got want need make made go going know see seen
take used using let lets like said says read your our their my me us him her who what when where
why how all each both few other another up down
have has had do does did being
""".split()
)

_PUNCT_RE = re.compile(r"[\s，。！？；：、“”‘’（）()\[\]【】《》<>·,.!?;:\"'`~@#$%^&*_+=|\\/—\-]+")

CUE_WORDS = [
    "重点",
    "难点",
    "注意",
    "关键",
    "定义",
    "公式",
    "结论",
    "因此",
    "所以",
    "总结",
    "首先",
    "其次",
    "最后",
    "举例",
    "例如",
    "记住",
    "考试",
    "作业",
    "步骤",
    "原理",
    "区别",
    "对比",
    "本质",
    "核心",
    "过程",
    "目标",
    "衡量",
    "更新",
    "收敛",
    "泛化",
    "结构",
    "区别",
    "步骤",
    "回顾",
    "important",
    "key",
    "remember",
    "conclusion",
    "definition",
    "formula",
    "therefore",
    "summary",
    "steps",
    "for example",
    "difference",
    "principle",
    "however",
]


def clean_text(text: str) -> str:
    return _PUNCT_RE.sub(" ", text or "").strip()


def tokenize(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    tokens = []
    for tok in jieba.lcut(cleaned):
        tok = tok.strip()
        if len(tok) < 2 or tok.isdigit():
            continue
        if tok.lower() in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+", text or "")
    return [p.strip() for p in parts if p.strip()]


def keyword_scores(texts: list[str], topk: int = 12) -> dict[str, float]:
    """A small TF-IDF implementation, no sklearn dependency needed."""
    doc_tokens = [tokenize(t) for t in texts]
    df: Counter[str] = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    n_docs = max(1, len(doc_tokens))
    tf: Counter[str] = Counter()
    for toks in doc_tokens:
        tf.update(toks)
    scores: dict[str, float] = {}
    for term, freq in tf.items():
        idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
        scores[term] = freq * idf
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    if not ranked:
        return {}
    maximum = max(v for _, v in ranked) or 1.0
    return {k: v / maximum for k, v in ranked}


def text_richness(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    length_score = min(1.0, len(tokens) / 40.0)
    return 0.55 * unique_ratio + 0.45 * length_score


def cue_word_score(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(text.count(w) for w in CUE_WORDS)
    return min(1.0, hits / 3.0)


def keyword_hit_ratio(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw and kw in text)
    return min(1.0, hits / max(1, min(len(keywords), 6)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


FILLERS = [
    "同学们好，",
    "大家好，",
    "这节课我们",
    "这节课",
    "接下来",
    "最后",
    "首先",
    "那么",
    "好，",
    "now",
    "so",
    "well",
    "okay",
]


def best_title(text: str, keywords: list[str], limit: int = 24) -> str:
    raw_units = [
        unit
        for sentence in split_sentences(text)
        for unit in re.split(r"(?<=[，,])\s*", sentence)
    ]
    candidates: list[tuple[float, int, str]] = []
    keyword_set = {
        keyword.lower()
        for keyword in keywords
        if keyword and keyword.lower() not in STOPWORDS
    }
    for index, raw in enumerate(raw_units):
        base = raw.strip("，。；：,.;:!?！？ ")
        lower = base.lower()
        for filler in FILLERS:
            if lower.startswith(filler):
                base = base[len(filler) :].lstrip(" ，,：:")
                break
        base = re.sub(r"^(?:and|but|so|then)\s+", "", base, flags=re.IGNORECASE)
        base = base.strip("，。；：,.;:!?！？ ")
        meaningful = tokenize(base)
        if not meaningful:
            continue
        compact = "".join(base.split())
        if len(compact) < 4:
            continue
        words = re.findall(r"[A-Za-z]+", base.lower())
        if words and words[-1] in STOPWORDS:
            continue
        lower_base = base.lower()
        keyword_hits = sum(
            1 for keyword in keyword_set if keyword and keyword in lower_base
        )
        length_score = 1.0 - min(1.0, abs(len(base) - 20) / 40)
        position_score = max(0.0, 2.0 - index * 0.18)
        question_penalty = 1.5 if base.rstrip().endswith(("?", "？")) else 0.0
        opening_penalty = (
            1.2
            if re.match(
                r"^(?:i|you|we|they|he|she|it|that|this|there|here|have|has|do|does)\b",
                lower_base,
            )
            else 0.0
        )
        score = (
            keyword_hits * 1.2
            + cue_word_score(base)
            + length_score
            + position_score
            - question_penalty
            - opening_penalty
        )
        candidates.append((score, index, base))
    if not candidates:
        fallback = "、".join(keywords[:3]) or clean_text(text)[:limit]
        return fallback[:limit] + ("…" if len(fallback) > limit else "")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    base = candidates[0][2]
    if len(base) > limit:
        clipped = base[:limit]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        base = clipped.rstrip("，。；：,.;:!?！？ ") + "…"
    return base
