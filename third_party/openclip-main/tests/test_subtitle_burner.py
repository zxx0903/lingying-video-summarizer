import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from core.subtitle_burner import SubtitleBurner, SubtitlePreparationJob


def test_parse_srt_text_clips_overlapping_segments():
    burner = SubtitleBurner()
    text = (
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "[Host] Hello\n\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "World\n"
    )

    segments = burner._parse_srt_text(text)

    assert segments == [
        {
            "start": "00:00:00,000",
            "end": "00:00:01,500",
            "text": "[Host] Hello",
        },
        {
            "start": "00:00:01,500",
            "end": "00:00:03,000",
            "text": "World",
        },
    ]


def test_generate_ass_uses_cjk_font_and_strips_speaker_tags(monkeypatch):
    burner = SubtitleBurner()
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("Noto Sans CJK SC", "/tmp/fonts")),
    )

    ass = burner._generate_ass(
        [
            {
                "start": "00:00:00,000",
                "end": "00:00:02,000",
                "text": "[Host] >> 你好\n第二行",
            }
        ]
    )

    assert "Style: Original,Noto Sans CJK SC,80,&H0000FFFF" in ass
    assert ",10,10,150,1" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Original,,0,0,0,,你好 第二行" in ass


def test_generate_ass_keeps_dual_track_layout_when_translation_exists(monkeypatch):
    burner = SubtitleBurner()
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("DejaVu Sans", None)),
    )

    ass = burner._generate_ass(
        [
            {
                "start": "00:00:00,000",
                "end": "00:00:02,000",
                "text": "Original line",
            }
        ],
        translated=[
            {
                "start": "00:00:00,000",
                "end": "00:00:02,000",
                "text": "Translated line",
            }
        ],
    )

    assert "Style: Original,DejaVu Sans,80,&H00FFFFFF" in ass
    assert "Style: Translation,DejaVu Sans,80,&H0000FFFF" in ass
    assert ",10,10,50,1" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Translation,,0,0,0,,Translated line" in ass


def test_prepare_ass_for_clip_uses_existing_translated_sidecar(tmp_path, monkeypatch):
    burner = SubtitleBurner()
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("DejaVu Sans", None)),
    )
    original_srt = tmp_path / "original.srt"
    original_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nOriginal line\n", encoding="utf-8")
    translated_srt = tmp_path / "translated.srt"
    translated_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nTranslated line\n", encoding="utf-8")
    ass_path = tmp_path / "clip.ass"

    assert burner.prepare_ass_for_clip(original_srt, ass_path, "Simplified Chinese", translated_srt_path=translated_srt)
    ass = ass_path.read_text(encoding="utf-8")

    assert "Translated line" in ass


def test_prepare_ass_for_clip_persists_generated_translated_sidecar(tmp_path, monkeypatch):
    burner = SubtitleBurner()
    burner.client = object()
    monkeypatch.setattr(
        burner,
        "_translate_srt",
        lambda segments, target_lang: [{"start": segments[0]["start"], "end": segments[0]["end"], "text": "你好"}],
    )
    original_srt = tmp_path / "original.srt"
    original_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nOriginal line\n", encoding="utf-8")
    translated_srt = tmp_path / "translated.srt"
    ass_path = tmp_path / "clip.ass"

    assert burner.prepare_ass_for_clip(
        original_srt,
        ass_path,
        "Simplified Chinese",
        translated_output_path=translated_srt,
    )
    assert translated_srt.exists()
    assert "你好" in translated_srt.read_text(encoding="utf-8")


def test_build_ass_filter_value_includes_fontsdir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("DejaVu Sans", "/tmp/fonts")),
    )

    filter_value = SubtitleBurner.build_ass_filter_value(tmp_path / "clip.ass", language="en")

    assert filter_value == f"ass={tmp_path / 'clip.ass'}:fontsdir=/tmp/fonts"


def test_build_ass_filter_value_omits_fontsdir_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("Arial", None)),
    )

    filter_value = SubtitleBurner.build_ass_filter_value(tmp_path / "clip.ass", language="en")

    assert filter_value == f"ass={tmp_path / 'clip.ass'}"


def test_build_ass_filter_value_escapes_filter_special_chars(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("DejaVu Sans", "C:/Windows/Fonts")),
    )

    filter_value = SubtitleBurner.build_ass_filter_value(tmp_path / "clip.ass", language="en")

    assert "fontsdir=C\\:/Windows/Fonts" in filter_value


def test_font_family_from_path_uses_primary_family_name(monkeypatch):
    class CompletedProcess:
        returncode = 0
        stdout = "Heiti TC,黑體-繁,黒体-繁\n"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(),
    )

    family = SubtitleBurner._font_family_from_path("/System/Library/Fonts/STHeiti Light.ttc")

    assert family == "Heiti TC"


def test_generate_preview_image_renders_subtitles(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not installed")

    filters = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if " ass " not in filters.stdout:
        pytest.skip("ffmpeg was built without libass support")

    burner = SubtitleBurner()
    background_path = tmp_path / "background.png"
    baseline_video_path = tmp_path / "baseline.mp4"
    baseline_path = tmp_path / "baseline.png"
    preview_path = tmp_path / "preview.png"

    burner._create_preview_background(background_path)
    subprocess.run(
        [
            ffmpeg,
            "-loop", "1",
            "-i", str(background_path),
            "-t", "2",
            "-pix_fmt", "yuv420p",
            "-an",
            "-y", str(baseline_video_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-ss", "0.5",
            "-i", str(baseline_video_path),
            "-frames:v", "1",
            "-update", "1",
            "-y", str(baseline_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    ok = burner.generate_preview_image(
        preview_path,
        original_text="Preview subtitle",
    )

    assert ok is True
    assert baseline_path.exists()
    assert preview_path.exists()
    diff = ImageChops.difference(Image.open(baseline_path), Image.open(preview_path))
    assert diff.getbbox() is not None


def test_preferred_subtitle_path_for_clip_prefers_whisper_sidecar(tmp_path):
    mp4 = tmp_path / "rank_01_test.mp4"
    mp4.write_bytes(b"video")
    original_srt = tmp_path / "rank_01_test.srt"
    original_srt.write_text("original", encoding="utf-8")
    whisper_srt = tmp_path / "rank_01_test.whisper.srt"
    whisper_srt.write_text("whisper", encoding="utf-8")

    preferred = SubtitleBurner.preferred_subtitle_path_for_clip(mp4)

    assert preferred == whisper_srt


def test_burn_subtitles_for_clips_uses_whisper_sidecar_when_available(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    output_dir = tmp_path / "out"
    clips_dir.mkdir()
    mp4 = clips_dir / "rank_01_test.mp4"
    mp4.write_bytes(b"video")
    (clips_dir / "rank_01_test.srt").write_text("original", encoding="utf-8")
    whisper_srt = clips_dir / "rank_01_test.whisper.srt"
    whisper_srt.write_text("whisper", encoding="utf-8")

    burner = SubtitleBurner()
    used_subtitles = []

    def fake_process_clip(mp4_path, srt_path, output_path, subtitle_translation=None, **_kwargs):
        used_subtitles.append(Path(srt_path))
        return True

    monkeypatch.setattr(burner, "_process_clip", fake_process_clip)

    result = burner.burn_subtitles_for_clips(str(clips_dir), str(output_dir))

    assert result["success"] is True
    assert used_subtitles == [whisper_srt]


def test_prepare_ass_for_clips_preserves_order_when_parallel(tmp_path, monkeypatch):
    burner = SubtitleBurner(
        subtitle_translation_max_workers=2,
        subtitle_translation_launch_stagger_seconds=0,
    )
    burner.client = object()
    jobs = []
    for name in ["a", "b", "c"]:
        mp4 = tmp_path / f"{name}.mp4"
        srt = tmp_path / f"{name}.srt"
        ass = tmp_path / f"{name}.ass"
        mp4.write_bytes(b"video")
        srt.write_text("unused", encoding="utf-8")
        jobs.append(SubtitlePreparationJob(mp4=mp4, srt=srt, ass_path=ass))

    def fake_prepare_ass_for_clip(srt_path, ass_path, subtitle_translation=None, **_kwargs):
        if Path(srt_path).stem == "a":
            time.sleep(0.12)
        else:
            time.sleep(0.01)
        Path(ass_path).write_text(Path(srt_path).stem, encoding="utf-8")
        return True

    monkeypatch.setattr(burner, "prepare_ass_for_clip", fake_prepare_ass_for_clip)

    results = burner.prepare_ass_for_clips(jobs, "Simplified Chinese")

    assert [result.job.mp4.name for result in results] == ["a.mp4", "b.mp4", "c.mp4"]
    assert [result.ok for result in results] == [True, True, True]
    assert [job.ass_path.read_text(encoding="utf-8") for job in jobs] == ["a", "b", "c"]


def test_prepare_ass_for_clips_respects_worker_cap(tmp_path, monkeypatch):
    burner = SubtitleBurner(
        subtitle_translation_max_workers=2,
        subtitle_translation_launch_stagger_seconds=0,
    )
    burner.client = object()
    jobs = []
    for index in range(4):
        mp4 = tmp_path / f"clip_{index}.mp4"
        srt = tmp_path / f"clip_{index}.srt"
        ass = tmp_path / f"clip_{index}.ass"
        mp4.write_bytes(b"video")
        srt.write_text("unused", encoding="utf-8")
        jobs.append(SubtitlePreparationJob(mp4=mp4, srt=srt, ass_path=ass))

    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_prepare_ass_for_clip(_srt_path, ass_path, subtitle_translation=None, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        Path(ass_path).write_text("ok", encoding="utf-8")
        with lock:
            active -= 1
        return True

    monkeypatch.setattr(burner, "prepare_ass_for_clip", fake_prepare_ass_for_clip)

    results = burner.prepare_ass_for_clips(jobs, "Simplified Chinese")

    assert all(result.ok for result in results)
    assert max_active == 2


def test_prepare_ass_for_clips_serial_when_worker_count_is_one(tmp_path, monkeypatch):
    burner = SubtitleBurner(
        subtitle_translation_max_workers=1,
        subtitle_translation_launch_stagger_seconds=0,
    )
    burner.client = object()
    jobs = []
    for name in ["a", "b", "c"]:
        mp4 = tmp_path / f"{name}.mp4"
        srt = tmp_path / f"{name}.srt"
        ass = tmp_path / f"{name}.ass"
        mp4.write_bytes(b"video")
        srt.write_text("unused", encoding="utf-8")
        jobs.append(SubtitlePreparationJob(mp4=mp4, srt=srt, ass_path=ass))

    call_order = []

    def fake_prepare_ass_for_clip(srt_path, ass_path, subtitle_translation=None, **_kwargs):
        call_order.append(Path(srt_path).stem)
        Path(ass_path).write_text("ok", encoding="utf-8")
        return True

    monkeypatch.setattr(burner, "prepare_ass_for_clip", fake_prepare_ass_for_clip)

    results = burner.prepare_ass_for_clips(jobs, "Simplified Chinese")

    assert all(result.ok for result in results)
    assert call_order == ["a", "b", "c"]


def test_burn_subtitles_for_clips_parallel_translation_keeps_summary_shape(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    output_dir = tmp_path / "out"
    clips_dir.mkdir()
    for name in ["rank_01_a", "rank_02_b"]:
        (clips_dir / f"{name}.mp4").write_bytes(b"video")
        (clips_dir / f"{name}.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
            encoding="utf-8",
        )

    burner = SubtitleBurner(
        subtitle_translation_max_workers=2,
        subtitle_translation_launch_stagger_seconds=0,
    )
    burner.client = object()
    burned = []

    def fake_prepare_ass_for_clip(_srt_path, ass_path, subtitle_translation=None, **_kwargs):
        Path(ass_path).write_text("ass", encoding="utf-8")
        return True

    def fake_burn_ass(mp4_path, ass_path, output_path):
        burned.append((Path(mp4_path).name, Path(ass_path).name, Path(output_path).name))
        Path(output_path).write_bytes(b"burned")
        return True

    monkeypatch.setattr(burner, "prepare_ass_for_clip", fake_prepare_ass_for_clip)
    monkeypatch.setattr(burner, "_burn_ass", fake_burn_ass)

    result = burner.burn_subtitles_for_clips(
        str(clips_dir),
        str(output_dir),
        subtitle_translation="Simplified Chinese",
    )

    assert result["success"] is True
    assert result["total_clips"] == 2
    assert result["successful_clips"] == 2
    assert result["failed_clips"] == 0
    assert [clip["filename"] for clip in result["processed_clips"]] == [
        "rank_01_a.mp4",
        "rank_02_b.mp4",
    ]
    assert [item[0] for item in burned] == ["rank_01_a.mp4", "rank_02_b.mp4"]
    assert not any(output_dir.glob("*.ass"))


def test_burn_subtitles_for_clips_persists_translated_sidecars(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    output_dir = tmp_path / "out"
    clips_dir.mkdir()
    mp4 = clips_dir / "rank_01_a.mp4"
    mp4.write_bytes(b"video")
    (clips_dir / "rank_01_a.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        encoding="utf-8",
    )

    burner = SubtitleBurner(
        subtitle_translation_max_workers=2,
        subtitle_translation_launch_stagger_seconds=0,
    )
    burner.client = object()

    def fake_translate_srt(segments, _target_lang):
        return [
            {
                "start": segments[0]["start"],
                "end": segments[0]["end"],
                "text": "你好",
            }
        ]

    def fake_burn_ass(_mp4_path, _ass_path, output_path):
        Path(output_path).write_bytes(b"burned")
        return True

    monkeypatch.setattr(burner, "_translate_srt", fake_translate_srt)
    monkeypatch.setattr(burner, "_burn_ass", fake_burn_ass)

    result = burner.burn_subtitles_for_clips(
        str(clips_dir),
        str(output_dir),
        subtitle_translation="Simplified Chinese",
    )

    translated_srt = clips_dir / "rank_01_a.translated.srt"
    assert translated_srt.exists()
    assert "你好" in translated_srt.read_text(encoding="utf-8")
    assert result["processed_clips"][0]["translated_subtitle_filename"] == translated_srt.name


def test_parse_translation_json_accepts_direct_json_payload():
    burner = SubtitleBurner()

    parsed = burner._parse_translation_json(
        '[{"id": 1, "translation": "你好"}, {"id": 2, "translation": "世界"}]',
        expected_count=2,
    )

    assert parsed == ["你好", "世界"]


def test_parse_translation_json_accepts_fenced_json_payload():
    burner = SubtitleBurner()

    parsed = burner._parse_translation_json(
        '```json\n[{"id": 1, "translation": "Hello"}]\n```',
        expected_count=1,
    )

    assert parsed == ["Hello"]


def test_translate_srt_falls_back_to_none_when_json_payload_is_malformed():
    burner = SubtitleBurner(enable_llm=False)

    class FakeClient:
        def simple_chat(self, prompt, model=None, temperature=None):
            return '{"id": 1, "translation": "missing array wrapper"}'

    burner.client = FakeClient()

    translated = burner._translate_srt(
        [{"start": "00:00:00,000", "end": "00:00:01,000", "text": "Hello"}],
        "Simplified Chinese",
    )

    assert translated is None
