from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def frame_time(index: int, fps: float) -> float:
    return index / max(0.1, fps)


def _imread(path: str | Path):
    """Read images through bytes so Windows paths containing Chinese work."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        pass
    return cv2.imread(str(path))


def frame_stats(path: str | Path) -> dict:
    img = _imread(path)
    if img is None:
        return {
            "brightness": 0.0,
            "contrast": 0.0,
            "colorfulness": 0.0,
            "edge_density": 0.0,
            "gray": None,
        }
    small = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 64.0)
    contrast = min(1.0, contrast)

    b, g, r = cv2.split(small.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness = float(
        (np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
        / 180.0
    )
    colorfulness = min(1.0, colorfulness)

    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "edge_density": edge_density,
        "saturation": saturation,
        "gray": gray,
    }


def segment_visual_stats(
    frames: list[Path],
    fps: float,
    start: float,
    end: float,
    cache: dict[str, dict] | None = None,
) -> dict:
    cache = cache if cache is not None else {}
    indices = []
    for i, path in enumerate(frames):
        t = frame_time(i, fps)
        if start - 1e-6 <= t < end + 1e-6:
            indices.append(i)
    if not indices:
        return {
            "frames": 0,
            "brightness": 0.0,
            "contrast": 0.0,
            "colorfulness": 0.0,
            "edge_density": 0.0,
            "motion": 0.0,
            "scene_changes": 0,
        }

    stats = []
    grays = []
    for i in indices:
        key = str(frames[i])
        if key not in cache:
            cache[key] = frame_stats(frames[i])
        st = cache[key]
        stats.append(st)
        if st.get("gray") is not None:
            grays.append(st["gray"])

    motion = 0.0
    scene_changes = 0
    if len(grays) >= 2:
        diffs = []
        for a, b in zip(grays[:-1], grays[1:]):
            diff = float(np.mean(cv2.absdiff(a, b)) / 255.0)
            diffs.append(diff)
            if diff > 0.18:
                scene_changes += 1
        motion = float(np.mean(diffs))

    def avg(key: str) -> float:
        return float(np.mean([s.get(key, 0.0) for s in stats]))

    return {
        "frames": len(stats),
        "brightness": avg("brightness"),
        "contrast": avg("contrast"),
        "colorfulness": avg("colorfulness"),
        "edge_density": avg("edge_density"),
        "saturation": avg("saturation"),
        "motion": motion,
        "scene_changes": scene_changes,
    }


def detect_scene_boundaries(
    frames: list[Path],
    fps: float,
    threshold: float = 0.16,
) -> list[float]:
    """Detect slide/shot changes from downscaled grayscale frames."""
    if not frames:
        return [0.0]
    grays = []
    for p in frames:
        img = _imread(p)
        if img is None:
            continue
        small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    times = [0.0]
    for i in range(1, len(grays)):
        diff = float(np.mean(cv2.absdiff(grays[i - 1], grays[i])) / 255.0)
        if diff > threshold:
            times.append(i / max(0.1, fps))
    if len(times) == 1:
        times.append(len(frames) / max(0.1, fps))
    return times
