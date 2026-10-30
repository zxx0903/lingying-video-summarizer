#!/usr/bin/env python3
"""
Chinese transcript generation via a local FunASR Paraformer checkout.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.config import PARAFORMER_DEVICE, PARAFORMER_PROJECT_DIR

logger = logging.getLogger(__name__)


class ParaformerTranscriptProcessor:
    """Bridge to the external funasr-paraformer project."""

    REQUIRED_MODULES = ("numpy", "soundfile", "torch", "funasr", "modelscope")

    def __init__(
        self,
        project_dir: str = PARAFORMER_PROJECT_DIR,
        device: str = PARAFORMER_DEVICE,
    ):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.device = device
        # Keep the venv entrypoint path instead of resolving symlinks. Executing
        # the resolved base interpreter bypasses the virtualenv site-packages.
        self.current_python_bin = Path(sys.executable)
        self.project_python_candidates = [
            self.project_dir / ".venv" / "bin" / "python",
            self.project_dir / ".venv" / "Scripts" / "python.exe",
        ]
        self.python_bin = self._resolve_python_bin()
        self.transcribe_script = self.project_dir / "tools" / "transcribe_long_audio.py"
        self.json_to_srt_script = self.project_dir / "tools" / "funasr_json_to_srt.py"

    def _resolve_python_bin(self) -> Path:
        for candidate in self.project_python_candidates:
            if candidate.exists():
                missing_modules = self._missing_modules_for_python(candidate)
                if not missing_modules:
                    return candidate
                logger.warning(
                    "Ignoring Paraformer helper Python %s because required modules are missing: %s",
                    candidate,
                    ", ".join(missing_modules),
                )
        return self.current_python_bin

    @classmethod
    def _missing_current_env_modules(cls) -> list[str]:
        return [
            module_name
            for module_name in cls.REQUIRED_MODULES
            if importlib.util.find_spec(module_name) is None
        ]

    def _missing_modules_for_python(self, python_bin: Path) -> list[str]:
        python_bin = Path(python_bin)
        if self._same_python_bin(python_bin, self.current_python_bin):
            return self._missing_current_env_modules()

        probe_code = (
            "import importlib.util, json; "
            f"mods = {list(self.REQUIRED_MODULES)!r}; "
            "missing = [m for m in mods if importlib.util.find_spec(m) is None]; "
            "print(json.dumps(missing))"
        )

        try:
            proc = subprocess.run(
                [str(python_bin), "-c", probe_code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (FileNotFoundError, OSError):
            return list(self.REQUIRED_MODULES)

        if proc.returncode != 0:
            return list(self.REQUIRED_MODULES)

        try:
            missing_modules = json.loads((proc.stdout or "").strip() or "[]")
        except json.JSONDecodeError:
            return list(self.REQUIRED_MODULES)

        if isinstance(missing_modules, list):
            return [str(module_name) for module_name in missing_modules]
        return list(self.REQUIRED_MODULES)

    @staticmethod
    def _same_python_bin(lhs: Path, rhs: Path) -> bool:
        try:
            return lhs == rhs or lhs.samefile(rhs)
        except OSError:
            return lhs == rhs

    def availability_error(self) -> str | None:
        if not self.project_dir.exists():
            return (
                f"Paraformer project dir not found: {self.project_dir}. "
                "Place a compatible helper checkout there or set PARAFORMER_PROJECT_DIR."
            )
        if not self.transcribe_script.exists():
            return f"Paraformer transcription script not found: {self.transcribe_script}"
        if not self.json_to_srt_script.exists():
            return f"Paraformer JSON→SRT script not found: {self.json_to_srt_script}"
        if not self.python_bin.exists():
            return f"Paraformer Python not found: {self.python_bin}"
        missing_modules = self._missing_modules_for_python(self.python_bin)
        if missing_modules:
            missing = ", ".join(missing_modules)
            if self._same_python_bin(self.python_bin, self.current_python_bin):
                return (
                    f"Paraformer dependencies missing in the current environment: {missing}. "
                    "Run `uv sync --extra paraformer` or provide a helper checkout with its own .venv."
                )
            return (
                f"Paraformer dependencies missing in helper environment {self.python_bin}: {missing}. "
                "Repair that helper .venv or remove it so OpenClip can fall back to the current repo environment."
            )
        return None

    def is_available(self) -> bool:
        return self.availability_error() is None

    def transcribe_chinese_to_srt(self, media_path: str, output_dir: str | Path) -> tuple[str, dict[str, Any]]:
        error = self.availability_error()
        if error:
            raise RuntimeError(error)

        media_path = Path(media_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_srt = output_dir / f"{media_path.stem}.srt"

        with tempfile.TemporaryDirectory(prefix="paraformer_", dir=str(output_dir)) as tmp_dir:
            tmp_output_dir = Path(tmp_dir)
            self._run(
                [
                    str(self.python_bin),
                    str(self.transcribe_script),
                    str(media_path),
                    "--lang",
                    "zh",
                    "--device",
                    self.device,
                    "--disable-pbar",
                    "--cleanup-normalized",
                    "-o",
                    str(tmp_output_dir),
                ],
                step=f"Paraformer transcription for {media_path.name}",
            )

            output_json = self._find_output_json(tmp_output_dir, media_path.stem)
            payload = json.loads(output_json.read_text(encoding="utf-8"))

            self._run(
                [
                    str(self.python_bin),
                    str(self.json_to_srt_script),
                    str(output_json),
                    str(output_srt),
                ],
                step=f"Paraformer SRT conversion for {media_path.name}",
            )

        return str(output_srt), payload

    def _find_output_json(self, output_dir: Path, stem: str) -> Path:
        direct_match = output_dir / f"{stem}.json"
        if direct_match.exists():
            return direct_match

        candidates = sorted(
            path
            for path in output_dir.glob("*.json")
            if path.name != "summary.jsonl"
        )
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"Could not locate Paraformer JSON output under {output_dir}")

    def _run(self, cmd: list[str], step: str) -> None:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            return

        stderr_tail = (proc.stderr or "")[-1200:]
        stdout_tail = (proc.stdout or "")[-1200:]
        raise RuntimeError(
            f"{step} failed with exit code {proc.returncode}\n"
            f"stdout:\n{stdout_tail}\n"
            f"stderr:\n{stderr_tail}"
        )
