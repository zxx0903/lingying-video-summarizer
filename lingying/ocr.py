from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

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


def _ocr_image(
    image: np.ndarray,
    min_confidence: float = 0.5,
    max_chars: int = 1600,
) -> tuple[str, float]:
    result, _elapsed = _engine()(image)
    lines: list[tuple[tuple[float, float], str]] = []
    confidences: list[float] = []
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
        confidences.append(score)

    lines.sort(key=lambda item: (round(item[0][0] / 12), item[0][1]))
    # OCR boxes are usually visual lines rather than complete sentences. Joining
    # them with spaces keeps wrapped phrases intact for sentence-level summarising.
    text = " ".join(line for _anchor, line in lines)
    text = re.sub(r"\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+", "", text)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return text[:max_chars], confidence


def _screen_crops(image: np.ndarray) -> list[np.ndarray]:
    """Find probable bright presentation screens and upscale them for OCR."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cutoff = max(105.0, float(np.percentile(gray, 78)))
    _threshold, mask = cv2.threshold(
        gray, cutoff, 255, cv2.THRESH_BINARY
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    image_area = float(height * width)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            continue
        area_ratio = area / image_area
        if area_ratio < 0.025 or area_ratio > 0.92:
            continue
        if w / h < 1.15 or w < width * 0.18:
            continue
        fill_ratio = area / float(w * h)
        if fill_ratio < 0.45:
            continue
        brightness = float(np.mean(gray[y : y + h, x : x + w])) / 255.0
        candidates.append(
            (area_ratio * fill_ratio * (0.55 + 0.45 * brightness), (x, y, w, h))
        )

    # Projected screens are not always bright. Rectangle edges also provide a
    # useful fallback for dark-mode terminals and dim projectors.
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=1,
    )
    edge_contours, _edge_hierarchy = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in edge_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0 or w / h < 1.15 or w < width * 0.18:
            continue
        area_ratio = w * h / image_area
        if area_ratio < 0.025 or area_ratio > 0.75:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if not 4 <= len(polygon) <= 6:
            continue
        brightness = float(np.mean(gray[y : y + h, x : x + w])) / 255.0
        candidates.append(
            (area_ratio * (0.45 + 0.55 * brightness) * 1.1, (x, y, w, h))
        )

    crops: list[np.ndarray] = []
    candidates.sort(reverse=True)
    chosen_rects: list[tuple[int, int, int, int]] = []
    for _score, rect in candidates:
        x, y, w, h = rect
        duplicate = False
        for cx, cy, cw, ch in chosen_rects:
            intersection = max(
                0, min(x + w, cx + cw) - max(x, cx)
            ) * max(0, min(y + h, cy + ch) - max(y, cy))
            if intersection / max(1, min(w * h, cw * ch)) > 0.7:
                duplicate = True
                break
        if not duplicate:
            chosen_rects.append(rect)
        if len(chosen_rects) >= 2:
            break

    for x, y, w, h in chosen_rects:
        margin_x = int(w * 0.025)
        margin_y = int(h * 0.035)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width, x + w + margin_x)
        y2 = min(height, y + h + margin_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        scale = min(4.0, max(2.0, 1280.0 / max(1, crop.shape[1])))
        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            l_channel
        )
        enhanced = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0)
        crops.append(enhanced)
    return crops


def read_frame_text(
    frame_path: str | Path,
    min_confidence: float = 0.5,
    max_chars: int = 1600,
) -> str:
    """Recognize screen text, enhancing a projected-screen crop when needed."""
    data = np.fromfile(str(frame_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        return ""
    full_text, _full_confidence = _ocr_image(
        image, min_confidence=min_confidence, max_chars=max_chars
    )
    compact_length = len(re.sub(r"\W+", "", full_text))
    if compact_length >= 24:
        return full_text

    best_text = full_text
    for crop in _screen_crops(image):
        candidate, _confidence = _ocr_image(
            crop, min_confidence=min_confidence, max_chars=max_chars
        )
        if len(re.sub(r"\W+", "", candidate)) > compact_length:
            best_text = candidate
            compact_length = len(re.sub(r"\W+", "", candidate))
    return best_text


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
