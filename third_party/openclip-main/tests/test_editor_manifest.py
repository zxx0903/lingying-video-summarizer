import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from job_manager import JobManager

from core.editor.manifest import (
    build_manifest,
    discover_manifest_by_project_id,
    load_manifest,
    parse_timecode_to_seconds,
    reconcile_manifest,
    save_manifest,
    upsert_manifest,
)
from core.editor.models import EditorAssetRegistry, EditorClip, EditorManifest, EditorRecoveryState, TitleRecipe, SubtitleRecipe, CoverRecipe, utc_now_iso


@pytest.fixture()
def editor_result(tmp_path):
    project_root = tmp_path / "processed_videos" / "sample-video"
    clips_dir = project_root / "clips"
    post_dir = project_root / "clips_post_processed"
    clips_dir.mkdir(parents=True)
    post_dir.mkdir(parents=True)

    raw_clip = clips_dir / "rank_01_test_clip.mp4"
    raw_clip.write_bytes(b"raw")
    original_srt = clips_dir / "rank_01_test_clip.srt"
    original_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    whisper_srt = clips_dir / "rank_01_test_clip.whisper.srt"
    whisper_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello whisper\n", encoding="utf-8")
    translated_srt = clips_dir / "rank_01_test_clip.translated.srt"
    translated_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")
    composed_clip = post_dir / raw_clip.name
    composed_clip.write_bytes(b"composed")

    cover = project_root / "cover_rank_01_test_clip.jpg"
    cover.write_bytes(b"jpg")
    vertical_cover = project_root / "cover_rank_01_test_clip_vertical.jpg"
    vertical_cover.write_bytes(b"jpg")

    result = SimpleNamespace(
        video_info={"title": "Sample Video", "duration": 120.0},
        source_video_path=str(tmp_path / "source.mp4"),
        video_path=str(clips_dir / "sample-video_part01.mp4"),
        video_parts=[str(clips_dir / "sample-video_part01.mp4")],
        transcript_parts=[str(clips_dir / "sample-video_part01.srt")],
        part_offsets={"part01": 60.0},
        clip_generation={
            "success": True,
            "output_dir": str(clips_dir),
            "clips_info": [
                {
                    "rank": 1,
                    "title": "Test Clip",
                    "filename": raw_clip.name,
                    "subtitle_filename": original_srt.name,
                    "whisper_subtitle_filename": whisper_srt.name,
                    "translated_subtitle_filename": translated_srt.name,
                    "duration": 15.0,
                    "video_part": "part01",
                    "time_range": "00:00:10 - 00:00:25",
                    "original_time_range": "00:00:10 - 00:00:25",
                    "normalization_details": {"start": "unchanged", "end": "unchanged"},
                    "engagement_level": "high",
                    "why_engaging": "Strong payoff.",
                }
            ],
        },
        post_processing={
            "success": True,
            "output_dir": str(post_dir),
        },
        cover_generation={
            "success": True,
            "covers": [
                {
                    "rank": 1,
                    "title": "Test Clip",
                    "filename": cover.name,
                    "path": str(cover),
                    "vertical_filename": vertical_cover.name,
                    "vertical_path": str(vertical_cover),
                }
            ],
        },
    )
    return result, project_root


def test_manifest_build_and_roundtrip_preserves_asset_slots(editor_result):
    result, project_root = editor_result
    manifest_path = upsert_manifest(
        video_root_dir=project_root,
        result=result,
        title_style="gradient_3d",
        title_font_size=40,
        subtitle_translation="English",
        subtitle_style_preset="clean",
        subtitle_style_font_size="large",
        subtitle_style_vertical_position="middle",
        subtitle_style_bilingual_layout="bilingual",
        subtitle_style_background_style="solid_box",
        cover_text_location="center",
        cover_fill_color="yellow",
        cover_outline_color="black",
    )

    manifest = load_manifest(manifest_path)
    clip = manifest.clips[0]

    assert manifest.project_id
    assert manifest.source_video_path == result.source_video_path
    assert clip.clip_id
    assert clip.asset_registry.raw_clip.endswith("rank_01_test_clip.mp4")
    assert clip.asset_registry.current_composed_clip.endswith("rank_01_test_clip.mp4")
    assert clip.asset_registry.subtitle_sidecars["original"].endswith("rank_01_test_clip.srt")
    assert clip.asset_registry.subtitle_sidecars["whisper"].endswith("rank_01_test_clip.whisper.srt")
    assert clip.asset_registry.subtitle_sidecars["translated"].endswith("rank_01_test_clip.translated.srt")
    assert clip.asset_registry.horizontal_cover.endswith("cover_rank_01_test_clip.jpg")
    assert clip.asset_registry.vertical_cover.endswith("cover_rank_01_test_clip_vertical.jpg")
    assert clip.part_offset_seconds == 60.0
    assert clip.absolute_start_time == "00:01:10"
    assert clip.absolute_end_time == "00:01:25"
    assert clip.absolute_time_range == "00:01:10 - 00:01:25"
    assert clip.title_recipe.style == "gradient_3d"
    assert clip.subtitle_recipe.override_text is None
    assert clip.subtitle_recipe.style_bilingual_layout == "bilingual"
    assert clip.cover_recipe.text == "Test Clip"
    assert clip.recovery.last_good_assets["raw_clip"].endswith("rank_01_test_clip.mp4")

    second_path = upsert_manifest(
        video_root_dir=project_root,
        result=result,
        title_style="gradient_3d",
        title_font_size=40,
        subtitle_translation="English",
        subtitle_style_preset="clean",
        subtitle_style_font_size="large",
        subtitle_style_vertical_position="middle",
        subtitle_style_bilingual_layout="bilingual",
        subtitle_style_background_style="solid_box",
        cover_text_location="center",
        cover_fill_color="yellow",
        cover_outline_color="black",
    )
    second_manifest = load_manifest(second_path)
    assert second_manifest.project_id == manifest.project_id
    assert second_manifest.clips[0].clip_id == clip.clip_id


def test_manifest_roundtrip_preserves_translated_subtitle_sidecar(tmp_path):
    manifest = EditorManifest(
        project_id="project-1",
        schema_version=1,
        source_video_title="Sample",
        source_video_path="/tmp/source.mp4",
        source_video_duration=120.0,
        project_root=str(tmp_path / "processed_videos" / "sample"),
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        clips=[
            EditorClip(
                clip_id="clip-1",
                rank=1,
                title="Clip",
                video_part="part01",
                source_video_path="/tmp/source.mp4",
                source_video_duration=120.0,
                part_offset_seconds=0.0,
                part_duration_seconds=30.0,
                start_time="00:00:10",
                end_time="00:00:20",
                absolute_start_time="00:00:10",
                absolute_end_time="00:00:20",
                original_start_time="00:00:10",
                original_end_time="00:00:20",
                duration=10.0,
                time_range="00:00:10 - 00:00:20",
                original_time_range="00:00:10 - 00:00:20",
                absolute_time_range="00:00:10 - 00:00:20",
                asset_registry=EditorAssetRegistry(
                    raw_clip="/tmp/raw.mp4",
                    current_composed_clip="/tmp/current.mp4",
                    subtitle_sidecars={
                        "original": "/tmp/original.srt",
                        "translated": "/tmp/translated.srt",
                        "active": "/tmp/original.srt",
                    },
                ),
                title_recipe=TitleRecipe(text="Clip", style="gradient_3d", font_size=40),
                subtitle_recipe=SubtitleRecipe(),
                cover_recipe=CoverRecipe(text="Clip"),
                recovery=EditorRecoveryState(),
            ),
        ],
    )
    manifest_path = tmp_path / "editor_project.json"
    save_manifest(manifest, manifest_path)

    loaded = load_manifest(manifest_path)

    assert loaded.clips[0].asset_registry.subtitle_translated == "/tmp/translated.srt"


def test_manifest_upsert_treats_changed_clip_identity_as_new_clip(editor_result):
    result, project_root = editor_result
    manifest_path = upsert_manifest(
        video_root_dir=project_root,
        result=result,
        title_style="gradient_3d",
        title_font_size=40,
        subtitle_translation="English",
        subtitle_style_preset="clean",
        subtitle_style_font_size="large",
        subtitle_style_vertical_position="middle",
        subtitle_style_bilingual_layout="bilingual",
        subtitle_style_background_style="solid_box",
        cover_text_location="center",
        cover_fill_color="yellow",
        cover_outline_color="black",
    )
    original_manifest = load_manifest(manifest_path)
    original_clip = original_manifest.clips[0]
    original_clip.title_recipe.text = "Edited Title"
    original_clip.start_time = "00:00:11"
    original_clip.end_time = "00:00:24"
    save_manifest(original_manifest, manifest_path)

    updated_result = editor_result[0]
    updated_result.clip_generation["clips_info"][0]["title"] = "Different Clip"
    updated_result.clip_generation["clips_info"][0]["filename"] = "rank_01_different_clip.mp4"
    updated_result.clip_generation["clips_info"][0]["original_time_range"] = "00:00:30 - 00:00:44"
    updated_result.clip_generation["clips_info"][0]["time_range"] = "00:00:30 - 00:00:44"
    new_clip_path = project_root / "clips" / "rank_01_different_clip.mp4"
    new_clip_path.write_bytes(b"new-raw")

    refreshed_path = upsert_manifest(
        video_root_dir=project_root,
        result=updated_result,
        title_style="gradient_3d",
        title_font_size=40,
        subtitle_translation="English",
        subtitle_style_preset="clean",
        subtitle_style_font_size="large",
        subtitle_style_vertical_position="middle",
        subtitle_style_bilingual_layout="bilingual",
        subtitle_style_background_style="solid_box",
        cover_text_location="center",
        cover_fill_color="yellow",
        cover_outline_color="black",
    )
    refreshed_manifest = load_manifest(refreshed_path)
    refreshed_clip = refreshed_manifest.clips[0]

    assert refreshed_clip.clip_id != original_clip.clip_id
    assert refreshed_clip.title_recipe.text == "Different Clip"
    assert refreshed_clip.start_time == "00:00:30"
    assert refreshed_clip.end_time == "00:00:44"
    assert refreshed_clip.asset_registry.raw_clip.endswith("rank_01_different_clip.mp4")


def test_discover_manifest_by_project_id_prefers_latest_duplicate(tmp_path):
    projects_root = tmp_path / "processed_videos"
    older_root = projects_root / "sample_0"
    newer_root = projects_root / "sample"
    older_root.mkdir(parents=True)
    newer_root.mkdir(parents=True)

    older = {
        "project_id": "duplicate-project",
        "updated_at": "2026-04-24T09:00:00+00:00",
    }
    newer = {
        "project_id": "duplicate-project",
        "updated_at": "2026-04-24T14:00:00+00:00",
    }
    (older_root / "editor_project.json").write_text(json.dumps(older), encoding="utf-8")
    newer_path = newer_root / "editor_project.json"
    newer_path.write_text(json.dumps(newer), encoding="utf-8")

    assert discover_manifest_by_project_id(projects_root, "duplicate-project") == newer_path


@pytest.mark.parametrize(
    ("job_payload", "use_manager", "expected_state", "expect_pending", "expect_current_path", "expect_error_substring"),
    [
        ({"status": "pending", "current_step": "Queued"}, False, "pending", True, None, None),
        ({"status": "processing", "current_step": "Running"}, True, "recoverable", True, None, "Interrupted"),
        ({"status": "completed", "current_step": "Done"}, False, "clean", False, "rerendered.mp4", None),
        ({"status": "failed", "current_step": "Failed", "error": "boom"}, False, "failed", False, "good.mp4", "boom"),
        ({"status": "cancelled", "current_step": "Cancelled"}, False, "cancelled", False, "good.mp4", "cancelled"),
        (None, False, "stale_pending", False, None, "missing"),
    ],
)
def test_manifest_reconcile_truth_table(tmp_path, job_payload, use_manager, expected_state, expect_pending, expect_current_path, expect_error_substring):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    manifest = EditorManifest(
        project_id="project-1",
        schema_version=1,
        source_video_title="Sample",
        source_video_path="/tmp/source.mp4",
        source_video_duration=120.0,
        project_root=str(tmp_path / "processed_videos" / "sample"),
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        clips=[
            EditorClip(
                clip_id="clip-1",
                rank=1,
                title="Clip",
                video_part="part01",
                source_video_path="/tmp/source.mp4",
                source_video_duration=120.0,
                part_offset_seconds=0.0,
                part_duration_seconds=30.0,
                start_time="00:00:10",
                end_time="00:00:20",
                absolute_start_time="00:00:10",
                absolute_end_time="00:00:20",
                original_start_time="00:00:10",
                original_end_time="00:00:20",
                duration=10.0,
                time_range="00:00:10 - 00:00:20",
                original_time_range="00:00:10 - 00:00:20",
                absolute_time_range="00:00:10 - 00:00:20",
                asset_registry=EditorAssetRegistry(raw_clip="/tmp/raw.mp4", current_composed_clip="/tmp/current.mp4"),
                title_recipe=TitleRecipe(text="Clip", style="gradient_3d", font_size=40),
                subtitle_recipe=SubtitleRecipe(),
                cover_recipe=CoverRecipe(text="Clip"),
                recovery=EditorRecoveryState(
                    dirty=True,
                    pending_job_id="job-1",
                    pending_operation="subtitle",
                    pending_assets={"current_composed_clip": "/tmp/rerendered.mp4"},
                    last_good_assets={"current_composed_clip": "/tmp/good.mp4"},
                ),
            )
        ],
    )
    manifest_path = Path(manifest.project_root)
    manifest_path.mkdir(parents=True)
    save_manifest(manifest)

    manager = None
    if job_payload is not None:
        payload = {
            "id": "job-1",
            "video_source": "/tmp/source.mp4",
            "options": {},
            "progress": 0,
            "result": None,
            "error": job_payload.get("error"),
            "created_at": utc_now_iso(),
            "started_at": utc_now_iso(),
            "completed_at": utc_now_iso() if job_payload["status"] in {"completed", "failed", "cancelled"} else None,
            **job_payload,
        }
        (jobs_dir / "job-1.json").write_text(json.dumps(payload), encoding="utf-8")
        if use_manager:
            manager = JobManager(str(jobs_dir))

    changed = reconcile_manifest(manifest, job_manager=manager, jobs_dir=jobs_dir)
    clip = manifest.clips[0]

    assert changed is True
    assert clip.recovery.recovery_state == expected_state
    assert bool(clip.recovery.pending_job_id) is expect_pending
    if expect_current_path is not None:
        assert clip.asset_registry.current_composed_clip.endswith(expect_current_path)
    if expect_error_substring:
        assert expect_error_substring.lower() in (clip.recovery.last_error or "").lower()


def test_manifest_reconcile_prefers_completed_disk_job_over_stale_live_pending_job(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    manifest = EditorManifest(
        project_id="project-1",
        schema_version=1,
        source_video_title="Sample",
        source_video_path="/tmp/source.mp4",
        source_video_duration=120.0,
        project_root=str(tmp_path / "processed_videos" / "sample"),
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        clips=[
            EditorClip(
                clip_id="clip-1",
                rank=1,
                title="Clip",
                video_part="part01",
                source_video_path="/tmp/source.mp4",
                source_video_duration=120.0,
                part_offset_seconds=0.0,
                part_duration_seconds=30.0,
                start_time="00:00:10",
                end_time="00:00:20",
                absolute_start_time="00:00:10",
                absolute_end_time="00:00:20",
                original_start_time="00:00:10",
                original_end_time="00:00:20",
                duration=10.0,
                time_range="00:00:10 - 00:00:20",
                original_time_range="00:00:10 - 00:00:20",
                absolute_time_range="00:00:10 - 00:00:20",
                asset_registry=EditorAssetRegistry(raw_clip="/tmp/raw.mp4", current_composed_clip="/tmp/current.mp4"),
                title_recipe=TitleRecipe(text="Clip", style="gradient_3d", font_size=40),
                subtitle_recipe=SubtitleRecipe(),
                cover_recipe=CoverRecipe(text="Clip"),
                recovery=EditorRecoveryState(
                    dirty=True,
                    pending_job_id="job-1",
                    pending_operation="boundary",
                    pending_assets={"current_composed_clip": "/tmp/rerendered.mp4"},
                    last_good_assets={"current_composed_clip": "/tmp/good.mp4"},
                ),
            )
        ],
    )

    disk_payload = {
        "id": "job-1",
        "video_source": "/tmp/source.mp4",
        "options": {},
        "status": "completed",
        "current_step": "Done",
        "progress": 100,
        "result": {"current_composed_clip": "/tmp/rerendered.mp4"},
        "error": None,
        "created_at": utc_now_iso(),
        "started_at": utc_now_iso(),
        "completed_at": utc_now_iso(),
    }
    (jobs_dir / "job-1.json").write_text(json.dumps(disk_payload), encoding="utf-8")

    class StalePendingJob:
        def to_dict(self):
            return {
                "id": "job-1",
                "video_source": "/tmp/source.mp4",
                "options": {},
                "status": "pending",
                "current_step": "Queued",
                "progress": 0,
                "result": None,
                "error": None,
                "created_at": utc_now_iso(),
                "started_at": utc_now_iso(),
                "completed_at": None,
            }

    class StaleManager:
        def get_job(self, job_id):
            assert job_id == "job-1"
            return StalePendingJob()

    changed = reconcile_manifest(manifest, job_manager=StaleManager(), jobs_dir=jobs_dir)
    clip = manifest.clips[0]

    assert changed is True
    assert clip.recovery.pending_job_id is None
    assert clip.recovery.pending_operation is None
    assert clip.recovery.recovery_state == "clean"
    assert clip.asset_registry.current_composed_clip == "/tmp/rerendered.mp4"


def test_parse_timecode_to_seconds_accepts_hh_mm_ss_fractional():
    assert parse_timecode_to_seconds("00:01:02.500") == pytest.approx(62.5)
