from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import requests

from .config import PipelineConfig
from .text_features import (
    best_title,
    cue_word_score,
    jaccard,
    keyword_scores,
    split_sentences,
    tokenize,
)


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_outline(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        if title and end > start:
            out.append({"start": start, "end": end, "title": title})
    out.sort(key=lambda x: x["start"])
    return out


def _outline_index(t: float, outline: list[dict]) -> int | None:
    for i, item in enumerate(outline):
        if item["start"] - 0.5 <= t < item["end"] + 0.5:
            return i
    return None


def _summary_units(text: str, max_chars: int = 150) -> list[str]:
    """Split long ASR sentences into summary-sized, order-preserving units."""
    units: list[str] = []
    for sentence in split_sentences(text):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        clauses = [
            part.strip()
            for part in re.split(r"(?<=[，,；;：:])\s*", sentence)
            if part.strip()
        ]
        current = ""
        for clause in clauses:
            if len(clause) > max_chars and not current:
                words = clause.split()
                while words:
                    chunk = ""
                    while words and len(chunk) + len(words[0]) + 1 <= max_chars:
                        chunk = f"{chunk} {words.pop(0)}".strip()
                    if not chunk:
                        chunk = words.pop(0)[:max_chars]
                    units.append(chunk)
                continue
            candidate = f"{current} {clause}".strip()
            if current and len(candidate) > max_chars:
                units.append(current)
                current = clause
            else:
                current = candidate
        if current:
            units.append(current)
    return units


def _summary_points(
    text: str,
    title: str,
    keywords: list[str] | None = None,
    limit: int = 220,
) -> str:
    """Pick diverse, informative clauses within a strict length budget."""
    keywords = keywords or []
    priority = (
        "目的",
        "内容",
        "原理",
        "步骤",
        "注意",
        "要求",
        "方法",
        "结论",
        "定义",
        "操作",
        "接线",
        "读写",
    )
    technical = (
        "CPU",
        "RAM",
        "寄存器",
        "地址",
        "总线",
        "数据",
        "存储器",
        "片选",
        "开关",
        "脉冲",
        "三态门",
    )
    scored: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for index, sentence in enumerate(_summary_units(text)):
        sentence = sentence.strip(" ；;，,")
        sentence = re.sub(r"\s+(?=[\u4e00-\u9fff])", "", sentence)
        sentence = re.sub(r"(?<=[\u4e00-\u9fff])\s+", "", sentence)
        compact = "".join(sentence.split())
        if len(compact) < 5 or compact in seen:
            continue
        seen.add(compact)
        if sentence == title or sentence == title + "实验":
            continue
        if len(compact) < 30 and any(
            word in sentence for word in ("西工大", "学院", "姓名", "学号")
        ):
            continue
        if sentence.startswith(title):
            stripped = sentence[len(title) :].lstrip(" ：:；;，,")
            if len(stripped) >= 5:
                sentence = stripped
        score = min(2.0, len(compact) / 50.0)
        score += 3.0 if any(word in sentence for word in priority) else 0.0
        score += 1.0 if any(word in sentence for word in technical) else 0.0
        lower_sentence = sentence.lower()
        score += 1.4 * sum(
            1 for keyword in keywords[:6] if keyword.lower() in lower_sentence
        )
        score += 0.8 * cue_word_score(sentence)
        score += max(0.0, 0.5 - index * 0.05)
        scored.append((score, index, sentence))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    picked: list[tuple[float, int, str]] = []
    picked_tokens: list[set[str]] = []
    used = 0
    for item in ranked:
        sentence = item[2]
        tokens = set(tokenize(sentence))
        if any(jaccard(tokens, previous) > 0.72 for previous in picked_tokens):
            continue
        addition = len(sentence) + (1 if picked else 0)
        if picked and used + addition > limit:
            continue
        if not picked and len(sentence) > limit:
            sentence = sentence[:limit]
        picked.append((item[0], item[1], sentence))
        picked_tokens.append(tokens)
        used += len(sentence) + (1 if len(picked) > 1 else 0)
        if used >= limit * 0.85 or len(picked) >= 3:
            break
    picked.sort(key=lambda item: item[1])
    out: list[str] = []
    for _score, _index, sentence in picked:
        out.append(sentence)
    return "；".join(out)


def _chapter_title(text: str, keywords: list[str], index: int) -> str:
    headings = (
        ("注意事项", "实验注意事项"),
        ("报告册要求", "实验报告要求"),
        ("实验要求", "实验要求"),
        ("原理图", "实验原理图"),
        ("读存储器", "存储器读操作"),
        ("写存储器", "存储器写操作"),
        ("实验步骤", "实验操作步骤"),
        ("实验目的", "实验目的"),
        ("实验内容", "实验内容"),
        ("接线", "实验接线"),
    )
    for needle, title in headings:
        if needle in text:
            return title
    if not text:
        return f"第 {index} 部分"
    title = best_title(text, keywords[:4], limit=32)
    weak_starts = (
        "我们会了解",
        "我们会",
        "但",
        "然而",
        "所以",
        "甚至",
        "相反",
        "而且",
        "以及",
        "that ",
        "have ",
        "i ",
        "you ",
        "we ",
        "they ",
        "it ",
        "this ",
    )
    if (
        title.startswith(weak_starts)
        or len(title.strip("… ")) < 4
    ):
        keyword_title = "、".join(word for word in keywords[:3] if word)
        if keyword_title:
            title = keyword_title
    return title


def _group_tokens(group: list[dict]) -> set[str]:
    result: set[str] = set()
    for seg in group:
        result.update(seg.get("tokens", []))
        result.update(seg.get("keywords", []))
    return result


def _group_span(group: list[dict]) -> float:
    return group[-1]["end"] - group[0]["start"]


def _group_duration(group: list[dict]) -> float:
    return sum(max(0.0, seg["end"] - seg["start"]) for seg in group)


def _merge_groups(left: list[dict], right: list[dict]) -> list[dict]:
    return sorted(left + right, key=lambda seg: seg["start"])


def _group_selected(selected: list[dict], max_chapters: int) -> list[list[dict]]:
    """Build topic-oriented groups without treating every shot as a chapter."""
    if not selected:
        return []
    selected = sorted(selected, key=lambda seg: seg["start"])
    groups: list[list[dict]] = []
    current = [selected[0]]
    for seg in selected[1:]:
        gap = seg["start"] - current[-1]["end"]
        similarity = jaccard(_group_tokens(current), set(seg.get("tokens", [])))
        too_long = _group_duration(current) >= 120.0
        theme_break = len(current) >= 4 and similarity < 0.035
        if gap > 50.0 or too_long or theme_break:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
    groups.append(current)

    # A dense edited video may still leave one oversized group. Split it at its
    # largest temporal and thematic break.
    split_limit = 6
    while any(len(group) > split_limit for group in groups):
        rebuilt: list[list[dict]] = []
        for group in groups:
            if len(group) <= split_limit:
                rebuilt.append(group)
                continue
            candidates = []
            for i in range(1, len(group)):
                gap = group[i]["start"] - group[i - 1]["end"]
                similarity = jaccard(
                    set(group[i - 1].get("tokens", [])),
                    set(group[i].get("tokens", [])),
                )
                imbalance = abs(i - len(group) / 2)
                candidates.append((gap - 20 * similarity - 3 * imbalance, i))
            split_at = max(candidates)[1]
            rebuilt.extend([group[:split_at], group[split_at:]])
        groups = rebuilt

    max_chapters = max(1, int(max_chapters))
    while len(groups) > max_chapters:
        merge_options = []
        for i in range(len(groups) - 1):
            left, right = groups[i], groups[i + 1]
            gap = max(0.0, right[0]["start"] - left[-1]["end"])
            similarity = jaccard(_group_tokens(left), _group_tokens(right))
            cost = gap / 60.0 + (1.0 - similarity)
            merge_options.append((cost, i))
        _, merge_at = min(merge_options)
        groups[merge_at : merge_at + 2] = [
            _merge_groups(groups[merge_at], groups[merge_at + 1])
        ]
    return groups


def _outline_groups(selected: list[dict], outline: list[dict]) -> list[list[dict]]:
    buckets: dict[int, list[dict]] = {}
    for seg in selected:
        idx = seg.get("scene_id")
        if idx is None:
            idx = _outline_index((seg["start"] + seg["end"]) / 2.0, outline)
        idx = -1 if idx is None else int(idx)
        buckets.setdefault(idx, []).append(seg)
    return [
        group
        for _, group in sorted(
            buckets.items(), key=lambda kv: min(s["start"] for s in kv[1])
        )
    ]


def build_chapters(
    selected: list[dict],
    top_keywords: list[str],
    outline: list[dict] | None = None,
    max_chapters: int = 8,
) -> list[dict]:
    if not selected:
        return []
    titled_groups: list[tuple[str | None, list[dict]]] = []
    outline = outline or []
    if outline:
        for group in _outline_groups(selected, outline):
            idx = int(group[0].get("scene_id", -1))
            title = outline[idx]["title"] if 0 <= idx < len(outline) else None
            titled_groups.append((title, group))
    else:
        titled_groups = [(None, g) for g in _group_selected(selected, max_chapters)]

    chapters = []
    for i, (outline_title, group) in enumerate(titled_groups, 1):
        text = " ".join(
            g.get("semantic_text") or g.get("text", "") for g in group
        ).strip()
        kws = []
        for g in group:
            for k in g.get("keywords", []):
                if k not in kws:
                    kws.append(k)
        title = outline_title or _chapter_title(text, kws, i)
        snippet = _summary_points(text, title, kws)
        chapters.append(
            {
                "index": i,
                "title": title,
                "start": round(group[0]["start"], 2),
                "end": round(group[-1]["end"], 2),
                "time": f"{_fmt_time(group[0]['start'])} - {_fmt_time(group[-1]['end'])}",
                "keywords": kws[:5],
                "summary": snippet or "该章节以画面演示为主，建议结合关键帧浏览。",
                "segment_ids": [g["id"] for g in group],
                "score": round(sum(g["score"] for g in group) / len(group), 4),
            }
        )
    return chapters


def build_overview(
    duration: float,
    selected: list[dict],
    top_keywords: list[str],
    chapters: list[dict],
) -> str:
    selected_duration = sum(s["end"] - s["start"] for s in selected)
    coverage = selected_duration / max(1e-6, duration)
    first = chapters[0]["title"] if chapters else "课程导入"
    last = chapters[-1]["title"] if len(chapters) > 1 else ""
    tail = f"，最后收束于「{last}」" if last else ""
    intro = (
        f"视频总长 {_fmt_time(duration)}，系统从 {len(selected)} 个重点片段中提炼出 "
        f"{len(chapters)} 个章节，摘要时长 {_fmt_time(selected_duration)}，"
        f"约为原视频的 {coverage:.0%}。内容从「{first}」开始{tail}。"
    )
    details = []
    for chapter in chapters[:4]:
        point = str(chapter.get("summary", "")).strip()
        if point:
            details.append(f"{chapter['title']}：{point[:90]}")
    if not details and top_keywords:
        details.append("核心主题：" + "、".join(top_keywords[:6]))
    return intro + ("主要内容包括：" + "；".join(details) + "。" if details else "")


def maybe_llm_polish(
    overview: str,
    chapters: list[dict],
    config: PipelineConfig,
) -> tuple[str, str]:
    """Return (overview, mode). Falls back silently to the template."""
    if not config.llm_enabled:
        return overview, "template"
    api_key = os.environ.get(config.llm_api_key_env, "").strip()
    if not api_key:
        return overview, "template (no api key)"

    payload_chapters = [
        {"title": c["title"], "time": c["time"], "summary": c["summary"]}
        for c in chapters
    ]
    prompt = (
        "你是一名课程助教。请根据下面的章节信息，写一段 120 字以内的中文课程摘要，"
        "保留时间顺序和关键知识点，不要编造未出现的内容。只输出摘要正文。\n"
        + json.dumps(payload_chapters, ensure_ascii=False)
    )
    try:
        resp = requests.post(
            config.llm_base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        if text:
            return text, "llm"
    except Exception:
        pass
    return overview, "template (llm failed)"


def build_summary(
    duration: float,
    selected: list[dict],
    segment_texts: Iterable[str],
    config: PipelineConfig,
    outline: list[dict] | None = None,
) -> dict:
    kw_scores = keyword_scores([t for t in segment_texts if t], topk=12)
    top_keywords = list(kw_scores.keys())
    chapters = build_chapters(
        selected,
        top_keywords,
        outline=outline,
        max_chapters=config.max_chapters,
    )
    overview = build_overview(duration, selected, top_keywords, chapters)
    from .vlm import polish_summary_with_vlm

    overview, chapters, mode = polish_summary_with_vlm(
        overview, chapters, config
    )
    if mode not in {"vlm"}:
        overview, mode = maybe_llm_polish(overview, chapters, config)
    return {
        "overview": overview,
        "keywords": top_keywords[:10],
        "chapters": chapters,
        "mode": mode,
    }
