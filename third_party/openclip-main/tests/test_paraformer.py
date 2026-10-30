from pathlib import Path

import pytest

from core.transcript_generation_paraformer import ParaformerTranscriptProcessor


def test_availability_error_reports_missing_project_dir(tmp_path):
    processor = ParaformerTranscriptProcessor(project_dir=tmp_path / "missing-paraformer")

    assert "Paraformer project dir not found" in processor.availability_error()


def test_availability_uses_current_env_when_helper_scripts_exist(tmp_path, monkeypatch):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "transcribe_long_audio.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (tools_dir / "funasr_json_to_srt.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)
    monkeypatch.setattr(processor, "_missing_current_env_modules", lambda: [])

    assert processor.availability_error() is None


def test_availability_error_suggests_paraformer_extra_when_current_env_deps_missing(tmp_path, monkeypatch):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "transcribe_long_audio.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (tools_dir / "funasr_json_to_srt.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)
    monkeypatch.setattr(processor, "_missing_current_env_modules", lambda: ["funasr", "modelscope"])

    error = processor.availability_error()
    assert "uv sync --extra paraformer" in error
    assert "funasr, modelscope" in error


def test_resolve_python_bin_falls_back_when_helper_env_is_broken(tmp_path, monkeypatch):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "transcribe_long_audio.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (tools_dir / "funasr_json_to_srt.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    helper_python = tmp_path / ".venv" / "bin" / "python"
    helper_python.parent.mkdir(parents=True)
    helper_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    def fake_missing_modules(self, python_bin):
        if Path(python_bin) == helper_python.resolve():
            return ["numpy"]
        if Path(python_bin) == self.current_python_bin:
            return []
        return []

    monkeypatch.setattr(
        ParaformerTranscriptProcessor,
        "_missing_modules_for_python",
        fake_missing_modules,
    )

    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)

    assert processor.python_bin == processor.current_python_bin
    assert processor.availability_error() is None


def test_find_output_json_prefers_stem_matched_file(tmp_path):
    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)
    expected = tmp_path / "clip.json"
    expected.write_text("{}", encoding="utf-8")
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")

    assert processor._find_output_json(tmp_path, "clip") == expected


def test_find_output_json_accepts_single_generated_candidate(tmp_path):
    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)
    candidate = tmp_path / "generated.json"
    candidate.write_text("{}", encoding="utf-8")
    (tmp_path / "summary.jsonl").write_text("", encoding="utf-8")

    assert processor._find_output_json(tmp_path, "clip") == candidate


def test_find_output_json_rejects_ambiguous_candidates(tmp_path):
    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)
    (tmp_path / "first.json").write_text("{}", encoding="utf-8")
    (tmp_path / "second.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Could not locate Paraformer JSON output"):
        processor._find_output_json(tmp_path, "clip")
