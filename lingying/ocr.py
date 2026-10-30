from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .asr import Cue

_OCR_ENGINE = None
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class OcrSample:
    time: float
    text: str


def _engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _box_anchor(box: list) -> tuple[float, float]:
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return min(ys), min(xs)


def read_frame_text(
    frame_path: str | Path,
    min_confidence: float = 0.5,
    max_chars: int = 1600,
) -> str:
    """Recognize slide/screen text and preserve the approximate reading order."""
    result, _elapsed = _engine()(str(frame_path))
    lines: list[tuple[tuple[float, float], str]] = []
    seen: set[str] = set()
    for item in result or []:
        if len(item) < 3:
            continue
        box, raw_text, raw_score = item[0], item[1], item[2]
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        text = _WHITESPACE_RE.sub(" ", str(raw_text or "")).strip()
        compact = re.sub(r"[\W_]+", "", text)
        if score < min_confidence or len(compact) < 2 or compact in seen:
            continue
        seen.add(compact)
        lines.append((_box_anchor(box), text))

    lines.sort(key=lambda item: (round(item[0][0] / 12), item[0][1]))
    # OCR boxes are usually visual lines rather than complete sentences. Joining
    # them with spaces keeps wrapped phrases intact for sentence-level summarising.
    text = " ".join(line for _anchor, line in lines)
    text = re.sub(r"\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+", "", text)
    return text[:max_chars]


def _fingerprint(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def _same_slide(left: str, right: str) -> bool:
    a, b = _fingerprint(left), _fingerprint(right)
    if not a or not b:
        return False
    length_ratio = min(len(a), len(b)) / max(len(a), len(b))
    if (a in b or b in a) and length_ratio >= 0.7:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.78


def merge_ocr_samples(samples: list[OcrSample], duration: float) -> list[Cue]:
    """Merge repeated OCR results into slide-like cues and drop unstable frames."""
    runs: list[dict] = []
    for sample in sorted(samples, key=lambda item: item.time):
        if not sample.text:
            continue
        if runs and _same_slide(sample.text, runs[-1]["best_text"]):
            if len(sample.text) > len(runs[-1]["best_text"]):
                runs[-1]["best_text"] = sample.text
            continue
        runs.append(
            {
                "start": sample.time,
                "best_text": sample.text,
            }
        )

    cues: list[Cue] = []
    for index, run in enumerate(runs):
        start = max(0.0, float(run["start"]))
        if index + 1 < len(runs):
            end = float(runs[index + 1]["start"])
        else:
            end = duration
        end = min(duration, max(start + 0.5, end))
        if end <= start:
            continue
        cues.append(Cue(start=start, end=end, text=str(run["best_text"])))
    return cues


def ocr_frames(
    frames: list[Path],
    fps: float,
    duration: float,
    max_frames: int = 80,
    min_confidence: float = 0.5,
) -> list[Cue]:
    """OCR an evenly sampled subset of extracted frames."""
    if not frames:
        return []
    max_frames = max(1, int(max_frames))
    if len(frames) <= max_frames:
        indices = list(range(len(frames)))
    elif max_frames == 1:
        indices = [len(frames) // 2]
    else:
        indices = sorted(
            {
                round(i * (len(frames) - 1) / (max_frames - 1))
                for i in range(max_frames)
            }
        )

    samples: list[OcrSample] = []
    for index in indices:
        samples.append(
            OcrSample(
                time=index / max(0.1, fps),
                text=read_frame_text(
                    frames[index],
                    min_confidence=min_confidence,
                ),
            )
        )
    return merge_ocr_samples(samples, duration)
