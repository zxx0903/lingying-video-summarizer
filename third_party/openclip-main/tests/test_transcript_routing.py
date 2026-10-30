import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.transcript_generation_whisper as transcript_module
from core.transcript_generation_whisper import (
    TranscriptProcessor,
    build_whisper_initial_prompt,
    select_transcript_backend,
    summarize_transcript_sources,
)


def test_select_transcript_backend_prefers_paraformer_for_chinese():
    assert select_transcript_backend("zh", paraformer_available=True, use_whisperx=False) == "paraformer"


def test_select_transcript_backend_uses_whisper_for_english_without_diarization():
    assert select_transcript_backend("en", paraformer_available=True, use_whisperx=False) == "whisper"


def test_select_transcript_backend_uses_whisperx_for_english_with_diarization():
    assert select_transcript_backend("en", paraformer_available=True, use_whisperx=True) == "whisperx"


def test_select_transcript_backend_falls_back_when_paraformer_is_unavailable():
    assert select_transcript_backend("zh", paraformer_available=False, use_whisperx=False) == "whisper"


def test_summarize_transcript_sources_handles_mixed_sources():
    assert summarize_transcript_sources(["paraformer", "whisper", "paraformer"]) == "mixed:paraformer,whisper"


def test_build_whisper_initial_prompt_prefers_simplified_chinese_for_zh():
    assert build_whisper_initial_prompt("zh") == "以下是普通话的简体中文字幕。"


def test_build_whisper_initial_prompt_ignores_non_chinese_languages():
    assert build_whisper_initial_prompt("en") is None


class FakeParaformerProcessor:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def is_available(self) -> bool:
        return True

    def availability_error(self):
        return None

    def transcribe_chinese_to_srt(self, media_path: str, output_dir: str):
        if self.should_fail:
            raise RuntimeError("paraformer failed")
        srt_path = Path(output_dir) / f"{Path(media_path).stem}.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n",
            encoding="utf-8",
        )
        return str(srt_path), {"segments": []}


def test_generate_routed_transcripts_uses_paraformer_for_chinese(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")

    processor = TranscriptProcessor()
    processor.paraformer_processor = FakeParaformerProcessor()
    monkeypatch.setattr(processor, "_detect_transcript_language", lambda _: "zh")

    result = asyncio.run(processor._generate_routed_transcripts(str(video_path), None))

    assert result["source"] == "paraformer"
    assert result["transcript_path"] == str(tmp_path / "clip.srt")
    assert result["transcript_parts"] == [str(tmp_path / "clip.srt")]


def test_generate_routed_transcripts_falls_back_to_whisper_when_paraformer_fails(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    whisper_calls = []

    def fake_run_whisper_cli(file_path, model_name, language, output_format, output_dir):
        whisper_calls.append(
            {
                "file_path": file_path,
                "model_name": model_name,
                "language": language,
                "output_format": output_format,
                "output_dir": output_dir,
            }
        )
        srt_path = Path(output_dir) / f"{Path(file_path).stem}.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nfallback\n",
            encoding="utf-8",
        )
        return True

    processor = TranscriptProcessor()
    processor.paraformer_processor = FakeParaformerProcessor(should_fail=True)
    monkeypatch.setattr(processor, "_detect_transcript_language", lambda _: "zh")
    monkeypatch.setattr(transcript_module, "run_whisper_cli", fake_run_whisper_cli)

    result = asyncio.run(processor._generate_routed_transcripts(str(video_path), None))

    assert result["source"] == "whisper_fallback"
    assert result["transcript_path"] == str(tmp_path / "clip.srt")
    assert whisper_calls == [
        {
            "file_path": str(video_path),
            "model_name": processor.whisper_model,
            "language": "zh",
            "output_format": "srt",
            "output_dir": str(tmp_path),
        }
    ]


def test_generate_routed_transcripts_summarizes_mixed_sources(tmp_path, monkeypatch):
    zh_video = tmp_path / "clip_zh.mp4"
    en_video = tmp_path / "clip_en.mp4"
    zh_video.write_bytes(b"fake-zh-video")
    en_video.write_bytes(b"fake-en-video")

    def fake_detect_language(media_path: str) -> str:
        return "zh" if media_path.endswith("clip_zh.mp4") else "en"

    def fake_run_whisper_cli(file_path, model_name, language, output_format, output_dir):
        srt_path = Path(output_dir) / f"{Path(file_path).stem}.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nenglish\n",
            encoding="utf-8",
        )
        return True

    processor = TranscriptProcessor()
    processor.paraformer_processor = FakeParaformerProcessor()
    monkeypatch.setattr(processor, "_detect_transcript_language", fake_detect_language)
    monkeypatch.setattr(transcript_module, "run_whisper_cli", fake_run_whisper_cli)

    result = asyncio.run(
        processor._generate_routed_transcripts(
            [str(zh_video), str(en_video)],
            None,
        )
    )

    assert result["source"] == "mixed:paraformer,whisper"
    assert result["transcript_path"] == ""
    assert result["transcript_parts"] == [
        str(tmp_path / "clip_zh.srt"),
        str(tmp_path / "clip_en.srt"),
    ]
