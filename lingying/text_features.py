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
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*", text or "")
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


FILLERS = ["同学们好，", "大家好，", "这节课我们", "这节课", "接下来", "最后", "首先", "那么", "好，"]


def best_title(text: str, keywords: list[str], limit: int = 24) -> str:
    sentences = split_sentences(text)
    base = sentences[0] if sentences else (text or "").strip()
    for filler in FILLERS:
        if base.startswith(filler):
            base = base[len(filler) :]
            break
    clauses = [c.strip("，。；：,.;: ") for c in re.split(r"[，。；;]", base) if c.strip()]
    if not clauses:
        base = clean_text(text)[:limit]
    else:
        base = clauses[0]
        if base.startswith(("按照", "如果", "当", "对于", "在")) and len(clauses) > 1:
            base = clauses[1]
    base = base.strip("，。；：,.;: ")
    if len(base) < 5 and keywords:
        base = f"{keywords[0]}：{base}"
    return base[:limit] + ("…" if len(base) > limit else "")
