from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return sr, data


def segment_audio_stats(
    samples: np.ndarray, sr: int, start: float, end: float
) -> dict:
    a = max(0, int(start * sr))
    b = min(len(samples), int(end * sr))
    if b <= a:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "zcr": 0.0,
            "silence_ratio": 1.0,
            "energy_var": 0.0,
            "voiced_ratio": 0.0,
        }
    chunk = samples[a:b]
    frame = max(1, int(sr * 0.02))
    n = len(chunk) // frame
    if n == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "zcr": 0.0,
            "silence_ratio": 1.0,
            "energy_var": 0.0,
            "voiced_ratio": 0.0,
        }
    trimmed = chunk[: n * frame].reshape(n, frame)
    rms_frames = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-9)
    threshold = max(0.008, float(np.percentile(rms_frames, 20)) * 0.6)
    voiced = rms_frames > threshold
    signs = np.signbit(trimmed)
    zcr = float(np.mean(np.abs(np.diff(signs.astype(np.int8), axis=1))))
    return {
        "rms": float(np.mean(rms_frames)),
        "peak": float(np.max(np.abs(chunk))),
        "zcr": zcr,
        "silence_ratio": float(1.0 - np.mean(voiced)),
        "energy_var": float(np.std(rms_frames)),
        "voiced_ratio": float(np.mean(voiced)),
    }
