from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Weights


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def normalize_segments(segments: list[dict]) -> None:
    """Attach normalized 0-1 features used by the fusion scorer."""
    text_rich = _minmax([s["text_stats"].get("richness", 0.0) for s in segments])
    text_cue = _minmax([s["text_stats"].get("cue", 0.0) for s in segments])
    text_kw = _minmax([s["text_stats"].get("keyword_hit", 0.0) for s in segments])

    audio_rms = _minmax([s["audio_stats"].get("rms", 0.0) for s in segments])
    audio_voiced = _minmax([s["audio_stats"].get("voiced_ratio", 0.0) for s in segments])
    audio_var = _minmax([s["audio_stats"].get("energy_var", 0.0) for s in segments])
    audio_rate = _minmax([s["text_stats"].get("speech_rate", 0.0) for s in segments])

    vis_edge = _minmax([s["visual_stats"].get("edge_density", 0.0) for s in segments])
    vis_motion = _minmax([s["visual_stats"].get("motion", 0.0) for s in segments])
    vis_color = _minmax([s["visual_stats"].get("colorfulness", 0.0) for s in segments])

    for i, seg in enumerate(segments):
        seg["norm"] = {
            "text_richness": text_rich[i],
            "text_cue": text_cue[i],
            "text_keyword": text_kw[i],
            "audio_rms": audio_rms[i],
            "audio_voiced": audio_voiced[i],
            "audio_var": audio_var[i],
            "audio_rate": audio_rate[i],
            "visual_edge": vis_edge[i],
            "visual_motion": vis_motion[i],
            "visual_color": vis_color[i],
        }


def reliability(seg: dict, has_audio: bool) -> tuple[float, float, float]:
    chars = seg["text_stats"].get("chars", 0)
    text_rel = min(1.0, chars / 80.0)
    rms = seg["audio_stats"].get("rms", 0.0)
    audio_rel = 1.0 if has_audio and rms > 0.004 else 0.18
    frames = seg["visual_stats"].get("frames", 0)
    visual_rel = 1.0 if frames >= 2 else 0.15
    return max(0.05, text_rel), audio_rel, visual_rel


def score_segments(
    segments: list[dict],
    weights: Weights | None = None,
    has_audio: bool = True,
) -> None:
    weights = (weights or Weights()).normalized()
    normalize_segments(segments)

    for seg in segments:
        n = seg["norm"]
        text_raw = (
            0.45 * n["text_keyword"]
            + 0.32 * n["text_cue"]
            + 0.23 * n["text_richness"]
        )
        audio_raw = (
            0.45 * n["audio_rms"]
            + 0.22 * n["audio_voiced"]
            + 0.18 * n["audio_var"]
            + 0.15 * n["audio_rate"]
        )
        visual_raw = (
            0.40 * n["visual_edge"]
            + 0.35 * n["visual_motion"]
            + 0.25 * n["visual_color"]
        )

        tr, ar, vr = reliability(seg, has_audio)
        w_text = weights.text * tr
        w_audio = weights.audio * ar
        w_visual = weights.visual * vr
        total = w_text + w_audio + w_visual
        if total <= 1e-9:
            w_text, w_audio, w_visual, total = 1.0, 1.0, 1.0, 3.0

        score = (
            w_text / total * text_raw
            + w_audio / total * audio_raw
            + w_visual / total * visual_raw
        )

        seg["modality"] = {
            "text": round(text_raw, 4),
            "audio": round(audio_raw, 4),
            "visual": round(visual_raw, 4),
        }
        seg["gate"] = {
            "text": round(w_text / total, 4),
            "audio": round(w_audio / total, 4),
            "visual": round(w_visual / total, 4),
        }
        text = seg.get("text", "")
        bonus = 0.0
        if any(w in text for w in ("总结", "回顾", "作业")):
            bonus += 0.06
        if any(w in text for w in ("重点", "注意", "记住", "考点")):
            bonus += 0.03
        if any(w in text for w in ("定义", "公式", "原理", "本质")):
            bonus += 0.02
        seg["structure_bonus"] = round(min(0.08, bonus), 4)
        seg["score"] = round(min(1.0, float(score) + seg["structure_bonus"]), 4)

    # A small length prior: extremely short or extremely long segments are less useful.
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur < 8:
            seg["score"] = round(seg["score"] * 0.85, 4)
        elif dur > 60:
            seg["score"] = round(seg["score"] * 0.92, 4)


def apply_template_prior(segments: list[dict], keywords: list[str]) -> None:
    """Apply a small, inspectable prior for a selected scenario template."""
    keywords = [word.strip() for word in keywords if word and word.strip()]
    if not keywords:
        for seg in segments:
            seg["template_hits"] = []
            seg["template_bonus"] = 0.0
        return

    for seg in segments:
        text = (
            seg.get("semantic_text") or seg.get("text", "") or ""
        ).lower()
        hits = [word for word in keywords if word.lower() in text]
        # Hitting two different template cues is materially stronger than one
        # generic word. Cap the bonus so evidence from the video still dominates.
        bonus = min(0.10, 0.035 * len(hits))
        seg["template_hits"] = hits
        seg["template_bonus"] = round(bonus, 4)
        seg["score"] = round(min(1.0, float(seg.get("score", 0.0)) + bonus), 4)


def apply_feedback(
    segments: list[dict],
    labels: dict[str, int],
    weights: Weights | None = None,
) -> Weights:
    """Adjust modality weights from user keep/drop decisions.

    A lightweight, explainable update: kept segments vote for the modality that
    contributed most; dropped segments vote against it.
    """
    base = (weights or Weights()).normalized()
    delta = {"text": 0.0, "audio": 0.0, "visual": 0.0}
    for seg in segments:
        label = labels.get(str(seg["id"]))
        if label is None:
            continue
        gate = seg.get("gate", {})
        dominant = max(gate, key=gate.get) if gate else "text"
        delta[dominant] += 0.05 if label else -0.04
    out = Weights(
        text=max(0.1, base.text + delta["text"]),
        audio=max(0.05, base.audio + delta["audio"]),
        visual=max(0.05, base.visual + delta["visual"]),
    ).normalized()
    return out


def train_reranker(segments: list[dict], labels: dict[str, int]):
    """Optional logistic reranker. Returns None when there is not enough data."""
    xs, ys = [], []
    for seg in segments:
        y = labels.get(str(seg["id"]))
        if y is None:
            continue
        m = seg.get("modality", {})
        dur = seg["end"] - seg["start"]
        xs.append(
            [
                m.get("text", 0.0),
                m.get("audio", 0.0),
                m.get("visual", 0.0),
                seg["score"],
                min(1.0, dur / 60.0),
                len(seg.get("keywords", [])) / 5.0,
            ]
        )
        ys.append(int(y))
    if len(set(ys)) < 2 or len(ys) < 8:
        return None
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(np.asarray(xs, dtype=float), np.asarray(ys))
        return model
    except Exception:
        return None


def save_weights(weights: Weights, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            {"text": weights.text, "audio": weights.audio, "visual": weights.visual},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
