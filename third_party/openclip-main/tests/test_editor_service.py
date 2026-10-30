import pytest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from core.editor.manifest import load_manifest, save_manifest, upsert_manifest
from core.editor.models import SubtitleSegment
from core.editor.service import EditorService, create_app



def _create_project(tmp_path):
    projects_root = tmp_path / "processed_videos"
    project_root = projects_root / "sample-video"
    clips_dir = project_root / "clips"
    post_dir = project_root / "clips_post_processed"
    splits_dir = project_root / "splits"
    clips_dir.mkdir(parents=True)
    post_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    raw_clip = clips_dir / "rank_01_test_clip.mp4"
    raw_clip.write_bytes(b"raw")
    (clips_dir / "rank_01_test_clip.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    (post_dir / raw_clip.name).write_bytes(b"composed")
    cover = project_root / "cover_rank_01_test_clip.jpg"
    cover.write_bytes(b"jpg")
    (project_root / "cover_rank_01_test_clip_vertical.jpg").write_bytes(b"jpg")

    (splits_dir / "sample-video_part01.mp4").write_bytes(b"part")
    (splits_dir / "sample-video_part01.srt").write_text("1\n00:00:10,000 --> 00:00:35,000\nHello\n", encoding="utf-8")

    result = SimpleNamespace(
        video_info={"title": "Sample Video", "duration": 120.0},
        source_video_path=str(tmp_path / "source.mp4"),
        video_path=str(tmp_path / "source.mp4"),
        video_parts=[str(splits_dir / "sample-video_part01.mp4")],
        transcript_parts=[str(splits_dir / "sample-video_part01.srt")],
        part_offsets={"part01": 60.0},
        clip_generation={
            "success": True,
            "output_dir": str(clips_dir),
            "clips_info": [
                {
                    "rank": 1,
                    "title": "Test Clip",
                    "filename": raw_clip.name,
                    "subtitle_filename": "rank_01_test_clip.srt",
                    "duration": 15.0,
                    "video_part": "part01",
                    "time_range": "00:00:10 - 00:00:25",
                    "original_time_range": "00:00:10 - 00:00:25",
                }
            ],
        },
        post_processing={"success": True, "output_dir": str(post_dir)},
        cover_generation={
            "success": True,
            "covers": [{"rank": 1, "title": "Test Clip", "filename": cover.name, "path": str(cover)}],
        },
    )
    manifest_path = upsert_manifest(
        video_root_dir=project_root,
        result=result,
        title_style="gradient_3d",
        title_font_size=40,
        subtitle_translation=None,
        subtitle_style_preset="default",
        subtitle_style_font_size="medium",
        subtitle_style_vertical_position="bottom",
        subtitle_style_bilingual_layout="auto",
        subtitle_style_background_style="none",
        cover_text_location="center",
        cover_fill_color="yellow",
        cover_outline_color="black",
    )
    manifest = load_manifest(manifest_path)
    return manifest, projects_root, tmp_path / "jobs"



def test_editor_service_load_update_and_rerender_contract(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)

    project = service.load_project(manifest.project_id)
    assert project["project_id"] == manifest.project_id
    assert project["active_clip_id"] == manifest.clips[0].clip_id

    clip_id = manifest.clips[0].clip_id
    updated_clip = service.update_clip_bounds(manifest.project_id, clip_id, "00:01:12", "00:01:27")
    assert updated_clip["time_range"] == "00:00:12 - 00:00:27"
    assert updated_clip["absolute_time_range"] == "00:01:12 - 00:01:27"
    assert updated_clip["recovery"]["cover_dirty"] is True

    updated_clip = service.update_clip_subtitles(
        manifest.project_id,
        clip_id,
        subtitle_segments=[
            {"start_time": "00:00:00,000", "end_time": "00:00:01,000", "text": "Edited subtitle"},
            {"start_time": "00:00:01,000", "end_time": "00:00:02,000", "text": "Follow-up line"},
        ],
    )
    assert updated_clip["subtitle_recipe"]["override_text"] == "Edited subtitle\nFollow-up line"
    assert updated_clip["subtitle_segments"][0]["text"] == "Edited subtitle"
    assert updated_clip["subtitle_segments"][1]["text"] == "Follow-up line"

    updated_clip = service.update_cover_title(manifest.project_id, clip_id, "New Cover Title")
    assert updated_clip["cover_recipe"]["text"] == "New Cover Title"

    rerender = service.request_rerender(manifest.project_id, clip_id, "subtitle")
    assert rerender["status"] == "pending"

    job_status = service.get_job_status(rerender["job_id"])
    assert job_status["status"] in {"pending", "processing", "completed", "failed"}
    assert job_status["options"]["clip_id"] == clip_id
    assert job_status["options"]["projects_root"] == str(projects_root.resolve())

    saved_manifest = load_manifest(Path(manifest.project_root) / "editor_project.json")
    saved_clip = saved_manifest.clip_by_id(clip_id)
    assert saved_clip.recovery.pending_job_id == rerender["job_id"]
    assert saved_clip.recovery.pending_operation == "subtitles"


def test_editor_service_resolves_repo_relative_media_paths_against_projects_root(tmp_path):
    projects_root = tmp_path / "processed_videos"
    source_path = projects_root / "sample" / "local_videos" / "source.mp4"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"video")

    service = EditorService(projects_root=projects_root, jobs_dir=tmp_path / "jobs")

    resolved = service._resolve_media_path("processed_videos/sample/local_videos/source.mp4")

    assert resolved == source_path



def test_editor_service_fastapi_routes(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    app = create_app(projects_root=projects_root, jobs_dir=jobs_dir)
    client = TestClient(app)
    clip_id = manifest.clips[0].clip_id

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    project = client.get(f"/api/projects/{manifest.project_id}")
    assert project.status_code == 200
    assert project.json()["project_id"] == manifest.project_id

    clip = client.get(f"/api/projects/{manifest.project_id}/clips/{clip_id}")
    assert clip.status_code == 200
    assert clip.json()["clip_id"] == clip_id
    assert clip.json()["effective_subtitle_text"] == 'Hello'
    assert clip.json()["absolute_time_range"] == "00:01:10 - 00:01:25"
    assert clip.json()["part_duration_seconds"] == 35.0

    bounds = client.patch(
        f"/api/projects/{manifest.project_id}/clips/{clip_id}/bounds",
        json={"start_time": "00:01:11", "end_time": "00:01:24", "speed": 1.5},
    )
    assert bounds.status_code == 200
    assert bounds.json()["time_range"] == "00:00:11 - 00:00:24"
    assert bounds.json()["absolute_time_range"] == "00:01:11 - 00:01:24"
    assert bounds.json()["speed"] == 1.5

    rerender = client.post(f"/api/projects/{manifest.project_id}/clips/{clip_id}/rerender/boundary")
    assert rerender.status_code == 200
    assert rerender.json()["operation"] == "boundary"

    subtitle = client.patch(
        f"/api/projects/{manifest.project_id}/clips/{clip_id}/subtitle",
        json={
            "subtitle_segments": [
                {"start_time": "00:00:00,000", "end_time": "00:00:01,000", "text": "First cue"},
                {"start_time": "00:00:01,000", "end_time": "00:00:02,000", "text": "Second cue"},
            ]
        },
    )
    assert subtitle.status_code == 200
    assert subtitle.json()["subtitle_segments"][0]["text"] == "First cue"
    assert subtitle.json()["subtitle_segments"][1]["text"] == "Second cue"


def test_update_clip_translated_subtitles_writes_sidecar_and_serializes_track(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / "editor_project.json"
    clip = manifest.clips[0]
    clip.subtitle_recipe.translation = "Simplified Chinese"
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    updated_clip = service.update_clip_translated_subtitles(
        manifest.project_id,
        clip.clip_id,
        subtitle_segments=[
            {"start_time": "00:00:00,000", "end_time": "00:00:01,000", "text": "你好"},
        ],
    )

    translated_path = Path(updated_clip["asset_registry"]["subtitle_sidecars"]["translated"])
    assert translated_path.exists()
    assert "你好" in translated_path.read_text(encoding="utf-8")
    assert updated_clip["has_translated_subtitles"] is True
    assert updated_clip["translated_subtitle_segments"][0]["text"] == "你好"


def test_subtitle_worker_reuses_existing_translated_sidecar_without_llm(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / "editor_project.json"
    clip = manifest.clips[0]
    clip.metadata["title_overlay_enabled"] = False
    clip.subtitle_recipe.translation = "Simplified Chinese"
    translated_path = Path(manifest.project_root) / "editor_overrides" / f"{clip.clip_id}.translated.srt"
    translated_path.parent.mkdir(parents=True, exist_ok=True)
    translated_path.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    clip.asset_registry.subtitle_sidecars["translated"] = str(translated_path)
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    calls = {"translated_srt_path": None, "translated_output_path": None}

    def fake_process_clip(self, mp4, srt, output, subtitle_translation=None, translated_srt_path=None, translated_output_path=None):
        calls["translated_srt_path"] = str(translated_srt_path) if translated_srt_path else None
        calls["translated_output_path"] = str(translated_output_path) if translated_output_path else None
        Path(output).write_bytes(b"composed-updated")
        return True

    monkeypatch.setattr("core.subtitle_burner.SubtitleBurner._process_clip", fake_process_clip, raising=False)

    service._subtitle_worker(manifest_path, clip.clip_id, None, lambda *_args: None)

    assert calls["translated_srt_path"] == str(translated_path)
    assert calls["translated_output_path"] is None


def test_subtitle_worker_generates_initial_translated_sidecar_when_missing(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / "editor_project.json"
    clip = manifest.clips[0]
    clip.metadata["title_overlay_enabled"] = False
    clip.subtitle_recipe.translation = "Simplified Chinese"
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    calls = {"translated_output_path": None}

    def fake_process_clip(self, mp4, srt, output, subtitle_translation=None, translated_srt_path=None, translated_output_path=None):
        calls["translated_output_path"] = str(translated_output_path) if translated_output_path else None
        if translated_output_path:
            Path(translated_output_path).write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
        Path(output).write_bytes(b"composed-updated")
        return True

    monkeypatch.setattr("core.subtitle_burner.SubtitleBurner._process_clip", fake_process_clip, raising=False)

    service._subtitle_worker(manifest_path, clip.clip_id, None, lambda *_args: None)

    saved_manifest = load_manifest(manifest_path)
    saved_clip = saved_manifest.clip_by_id(clip.clip_id)
    assert calls["translated_output_path"] is not None
    assert saved_clip.asset_registry.subtitle_translated is not None
    assert Path(saved_clip.asset_registry.subtitle_translated).exists()


def test_build_subtitle_burner_enables_llm_for_translation_bootstrap_when_api_key_exists(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    clip = manifest.clips[0]
    clip.subtitle_recipe.translation = "Simplified Chinese"
    monkeypatch.setenv("QWEN_API_KEY", "test-qwen-key")

    captured_kwargs = {}

    def fake_init(self, **kwargs):
        captured_kwargs.update(kwargs)
        self.client = object() if kwargs.get("enable_llm") else None

    monkeypatch.setattr("core.editor.service.SubtitleBurner.__init__", fake_init)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    burner = service._build_subtitle_burner(clip, bootstrap_translation=True)

    assert burner.client is not None
    assert captured_kwargs["enable_llm"] is True
    assert captured_kwargs["provider"] == "qwen"


def test_boundary_worker_remaps_translated_sidecar_to_refreshed_timings(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / "editor_project.json"
    clip = manifest.clips[0]
    clip.metadata["title_overlay_enabled"] = False
    clip.subtitle_recipe.translation = "Simplified Chinese"
    clip.asset_registry.subtitle_sidecars["active"] = clip.asset_registry.subtitle_sidecars["original"]
    translated_path = Path(manifest.project_root) / "editor_overrides" / f"{clip.clip_id}.translated.srt"
    translated_path.parent.mkdir(parents=True, exist_ok=True)
    translated_path.write_text(
        "\n\n".join(
            [
                "1\n00:00:00,000 --> 00:00:01,000\n第一行",
                "2\n00:00:01,000 --> 00:00:02,000\n第二行",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    clip.asset_registry.subtitle_sidecars["translated"] = str(translated_path)
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)

    def fake_create_clip(self, source_video_path, start_time, end_time, output_path, title, speed=1.0):
        Path(output_path).write_bytes(b"raw-updated")
        return True

    def fake_extract_subtitle(self, subtitle_path, start_time, end_time, output_path, speed=1.0):
        Path(output_path).write_text(
            "\n\n".join(
                [
                    "1\n00:00:00,000 --> 00:00:01,200\nDerived one",
                    "2\n00:00:01,200 --> 00:00:02,400\nDerived two",
                ]
            ) + "\n",
            encoding="utf-8",
        )
        return True

    def fake_process_clip(self, mp4, srt, output, subtitle_translation=None, translated_srt_path=None, translated_output_path=None):
        Path(output).write_bytes(b"composed-updated")
        return True

    monkeypatch.setattr("core.clip_generator.ClipGenerator._create_clip", fake_create_clip, raising=False)
    monkeypatch.setattr("core.clip_generator.ClipGenerator._extract_subtitle_from_file", fake_extract_subtitle, raising=False)
    monkeypatch.setattr("core.subtitle_burner.SubtitleBurner._process_clip", fake_process_clip, raising=False)

    service._boundary_worker(manifest_path, clip.clip_id, None, lambda *_args: None)

    remapped_text = translated_path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,200" in remapped_text
    assert "00:00:01,200 --> 00:00:02,400" in remapped_text
    assert "第一行" in remapped_text
    assert "第二行" in remapped_text

def test_editor_spa_html_is_not_cached(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    app = create_app(projects_root=projects_root, jobs_dir=jobs_dir)
    client = TestClient(app)

    response = client.get(f"/projects/{manifest.project_id}")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"


def test_cover_title_update_does_not_mutate_title_overlay_recipe(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    clip_id = manifest.clips[0].clip_id

    updated = service.update_cover_title(manifest.project_id, clip_id, 'Cover Only Title')

    assert updated['cover_recipe']['text'] == 'Cover Only Title'
    assert updated['title_recipe']['text'] == 'Test Clip'
    assert updated['title'] == 'Test Clip'


def test_cover_worker_prefers_raw_clip_over_post_processed_clip(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / 'editor_project.json'
    clip = manifest.clips[0]
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    calls = {'source_clip': None}

    def fake_generate_cover(self, video_path, title_text, output_path, **_kwargs):
        calls['source_clip'] = video_path
        Path(output_path).write_bytes(b'jpg')
        Path(output_path.replace('.jpg', '_vertical.jpg')).write_bytes(b'jpg')
        return True

    monkeypatch.setattr('core.cover_image_generator.CoverImageGenerator.generate_cover', fake_generate_cover, raising=False)

    result = service._cover_worker(manifest_path, clip.clip_id, None, lambda *_args: None)

    assert result['horizontal_cover'].endswith('.jpg')
    assert calls['source_clip'] == clip.asset_registry.raw_clip


def test_subtitle_worker_preserves_original_generation_behavior_without_titles(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / 'editor_project.json'
    clip = manifest.clips[0]
    clip.metadata['title_overlay_enabled'] = False
    clip.subtitle_recipe.override_text = 'Edited subtitle'
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    calls = {'subtitle_only': 0, 'title_overlay': 0}

    def fake_process_clip(self, mp4, srt, output, subtitle_translation=None, **kwargs):
        calls['subtitle_only'] += 1
        Path(output).write_bytes(b'subtitle-only')
        return True

    def fake_add_title(self, *args, **kwargs):
        calls['title_overlay'] += 1
        return True

    monkeypatch.setattr('core.subtitle_burner.SubtitleBurner._process_clip', fake_process_clip, raising=False)
    monkeypatch.setattr('core.title_adder.TitleAdder._add_artistic_title', fake_add_title, raising=False)

    result = service._subtitle_worker(manifest_path, clip.clip_id, None, lambda *_args: None)

    assert result['current_composed_clip'].endswith('.mp4')
    assert calls['subtitle_only'] == 1
    assert calls['title_overlay'] == 0


def test_subtitle_worker_writes_override_segments_with_original_timings(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / 'editor_project.json'
    clip = manifest.clips[0]
    clip.metadata['title_overlay_enabled'] = False
    clip.subtitle_recipe.override_text = 'Edited subtitle\nFollow-up line'
    clip.subtitle_recipe.override_segments = [
        SubtitleSegment(start_time='00:00:00,000', end_time='00:00:01,200', text='Edited subtitle'),
        SubtitleSegment(start_time='00:00:01,200', end_time='00:00:02,500', text='Follow-up line'),
    ]
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    calls = {'subtitle_path': None, 'subtitle_content': None}

    def fake_process_clip(self, mp4, srt, output, subtitle_translation=None, **kwargs):
        calls['subtitle_path'] = str(srt)
        calls['subtitle_content'] = Path(srt).read_text(encoding='utf-8')
        Path(output).write_bytes(b'composed-updated')
        return True

    monkeypatch.setattr('core.subtitle_burner.SubtitleBurner._process_clip', fake_process_clip, raising=False)

    service._subtitle_worker(manifest_path, clip.clip_id, None, lambda *_args: None)

    assert calls['subtitle_path'] is not None
    assert "00:00:00,000 --> 00:00:01,200" in calls['subtitle_content']
    assert "00:00:01,200 --> 00:00:02,500" in calls['subtitle_content']
    assert "Edited subtitle" in calls['subtitle_content']
    assert "Follow-up line" in calls['subtitle_content']


def test_legacy_override_text_is_mapped_onto_derived_subtitle_timings(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    source_subtitle = tmp_path / "source_multi.srt"
    source_subtitle.write_text(
        "\n\n".join(
            [
                "1\n00:00:10,000 --> 00:00:11,000\nFirst original",
                "2\n00:00:11,000 --> 00:00:12,000\nSecond original",
                "3\n00:00:12,000 --> 00:00:13,000\nThird original",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    manifest_path = Path(manifest.project_root) / "editor_project.json"
    clip = manifest.clips[0]
    clip.metadata["source_subtitle_path"] = str(source_subtitle)
    clip.start_time = "00:00:10"
    clip.end_time = "00:00:13"
    clip.subtitle_recipe.override_text = "Line one\nLine two\nLine three"
    clip.subtitle_recipe.override_segments = []
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)

    serialized = service.get_clip(manifest.project_id, clip.clip_id)

    assert [segment["text"] for segment in serialized["subtitle_segments"]] == ["Line one", "Line two", "Line three"]
    assert serialized["subtitle_segments"][0]["start_time"] == "00:00:00,000"
    assert serialized["subtitle_segments"][1]["start_time"] == "00:00:01,000"
    assert serialized["subtitle_segments"][2]["start_time"] == "00:00:02,000"


def test_boundary_worker_refreshes_post_processed_clip_when_subtitles_are_derived(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / 'editor_project.json'
    clip = manifest.clips[0]
    clip.metadata['title_overlay_enabled'] = False
    clip.subtitle_recipe.override_text = None
    clip.asset_registry.subtitle_sidecars['active'] = clip.asset_registry.subtitle_sidecars['original']
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    calls = {'create_clip': 0, 'extract_subtitle': 0, 'subtitle_only': 0}

    def fake_create_clip(self, source_video_path, start_time, end_time, output_path, title, speed=1.0):
        calls['create_clip'] += 1
        calls['speed'] = speed
        Path(output_path).write_bytes(b'raw-updated')
        return True

    def fake_extract_subtitle(self, subtitle_path, start_time, end_time, output_path, speed=1.0):
        calls['extract_subtitle'] += 1
        calls['subtitle_speed'] = speed
        Path(output_path).write_text("1\n00:00:00,000 --> 00:00:01,000\nDerived\n", encoding='utf-8')
        return True

    def fake_process_clip(self, mp4, srt, output, subtitle_translation=None, **kwargs):
        calls['subtitle_only'] += 1
        Path(output).write_bytes(b'composed-updated')
        return True

    monkeypatch.setattr('core.clip_generator.ClipGenerator._create_clip', fake_create_clip, raising=False)
    monkeypatch.setattr('core.clip_generator.ClipGenerator._extract_subtitle_from_file', fake_extract_subtitle, raising=False)
    monkeypatch.setattr('core.subtitle_burner.SubtitleBurner._process_clip', fake_process_clip, raising=False)

    result = service._boundary_worker(manifest_path, clip.clip_id, None, lambda *_args: None)
    saved_manifest = load_manifest(manifest_path)
    saved_clip = saved_manifest.clip_by_id(clip.clip_id)

    assert calls['create_clip'] == 1
    assert calls['speed'] == 1.0
    assert calls['extract_subtitle'] >= 1
    assert calls['subtitle_speed'] == 1.0
    assert calls['subtitle_only'] == 1
    assert Path(result['current_composed_clip']).parent.name == 'clips_post_processed'
    assert Path(saved_clip.asset_registry.current_composed_clip).parent.name == 'clips_post_processed'


def test_boundary_worker_keeps_manual_override_on_raw_clip_until_subtitle_rerender(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / 'editor_project.json'
    clip = manifest.clips[0]
    clip.metadata['title_overlay_enabled'] = False
    clip.subtitle_recipe.override_text = 'Manual override text'
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    calls = {'create_clip': 0, 'subtitle_only': 0}

    def fake_create_clip(self, source_video_path, start_time, end_time, output_path, title, speed=1.0):
        calls['create_clip'] += 1
        Path(output_path).write_bytes(b'raw-updated')
        return True

    def fake_process_clip(self, mp4, srt, output, subtitle_translation=None, **kwargs):
        calls['subtitle_only'] += 1
        Path(output).write_bytes(b'composed-updated')
        return True

    monkeypatch.setattr('core.clip_generator.ClipGenerator._create_clip', fake_create_clip, raising=False)
    monkeypatch.setattr('core.subtitle_burner.SubtitleBurner._process_clip', fake_process_clip, raising=False)

    result = service._boundary_worker(manifest_path, clip.clip_id, None, lambda *_args: None)
    saved_manifest = load_manifest(manifest_path)
    saved_clip = saved_manifest.clip_by_id(clip.clip_id)

    assert calls['create_clip'] == 1
    assert calls['subtitle_only'] == 0
    assert result['current_composed_clip'] == saved_clip.asset_registry.raw_clip
    assert saved_clip.asset_registry.current_composed_clip == saved_clip.asset_registry.raw_clip


def test_boundary_worker_uses_selected_speed_for_raw_clip_and_subtitle_sidecars(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    manifest_path = Path(manifest.project_root) / 'editor_project.json'
    clip = manifest.clips[0]
    clip.speed = 2.0
    clip.metadata['title_overlay_enabled'] = False
    clip.subtitle_recipe.override_text = None
    clip.asset_registry.subtitle_sidecars['active'] = clip.asset_registry.subtitle_sidecars['original']
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    calls = {'create_speed': None, 'subtitle_speeds': []}

    def fake_create_clip(self, source_video_path, start_time, end_time, output_path, title, speed=1.0):
        calls['create_speed'] = speed
        Path(output_path).write_bytes(b'raw-updated')
        return True

    def fake_extract_subtitle(self, subtitle_path, start_time, end_time, output_path, speed=1.0):
        calls['subtitle_speeds'].append(speed)
        Path(output_path).write_text("1\n00:00:00,000 --> 00:00:00,500\nDerived\n", encoding='utf-8')
        return True

    def fake_process_clip(self, mp4, srt, output, subtitle_translation=None, **kwargs):
        Path(output).write_bytes(b'composed-updated')
        return True

    monkeypatch.setattr('core.clip_generator.ClipGenerator._create_clip', fake_create_clip, raising=False)
    monkeypatch.setattr('core.clip_generator.ClipGenerator._extract_subtitle_from_file', fake_extract_subtitle, raising=False)
    monkeypatch.setattr('core.subtitle_burner.SubtitleBurner._process_clip', fake_process_clip, raising=False)

    service._boundary_worker(manifest_path, clip.clip_id, None, lambda *_args: None)

    assert calls['create_speed'] == 2.0
    assert calls['subtitle_speeds']
    assert all(speed == 2.0 for speed in calls['subtitle_speeds'])


def test_derived_subtitle_segments_are_retimed_for_selected_speed(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    source_subtitle = tmp_path / "source_speed.srt"
    source_subtitle.write_text(
        "\n\n".join(
            [
                "1\n00:00:10,000 --> 00:00:12,000\nFirst original",
                "2\n00:00:12,000 --> 00:00:16,000\nSecond original",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    manifest_path = Path(manifest.project_root) / "editor_project.json"
    clip = manifest.clips[0]
    clip.metadata["source_subtitle_path"] = str(source_subtitle)
    clip.start_time = "00:00:10"
    clip.end_time = "00:00:16"
    clip.absolute_start_time = "00:01:10"
    clip.absolute_end_time = "00:01:16"
    clip.speed = 2.0
    clip.subtitle_recipe.override_text = None
    clip.subtitle_recipe.override_segments = []
    save_manifest(manifest, manifest_path)

    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)

    serialized = service.get_clip(manifest.project_id, clip.clip_id)

    assert serialized["subtitle_segments"][0]["start_time"] == "00:00:00,000"
    assert serialized["subtitle_segments"][0]["end_time"] == "00:00:01,000"
    assert serialized["subtitle_segments"][1]["start_time"] == "00:00:01,000"
    assert serialized["subtitle_segments"][1]["end_time"] == "00:00:03,000"


def test_request_rerender_rejects_duplicate_pending_job(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    clip_id = manifest.clips[0].clip_id

    service.request_rerender(manifest.project_id, clip_id, 'subtitle')

    with pytest.raises(ValueError, match='already has a pending rerender job'):
        service.request_rerender(manifest.project_id, clip_id, 'cover')


def test_boundary_rerender_queue_clears_manual_subtitle_override_immediately(tmp_path, monkeypatch):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    clip_id = manifest.clips[0].clip_id

    service.update_clip_subtitles(
        manifest.project_id,
        clip_id,
        subtitle_segments=[
            {"start_time": "00:00:00,000", "end_time": "00:00:01,000", "text": "Manual override text"},
        ],
    )
    service.update_clip_bounds(manifest.project_id, clip_id, "00:01:12", "00:01:27")

    monkeypatch.setattr(service.job_manager, "start_job", lambda *_args, **_kwargs: None)

    queued = service.request_rerender(manifest.project_id, clip_id, "boundary")
    assert queued["status"] == "pending"

    refreshed_clip = service.get_clip(manifest.project_id, clip_id)
    assert refreshed_clip["subtitle_recipe"]["override_text"] is None
    assert refreshed_clip["subtitle_recipe"]["override_segments"] == []
    assert refreshed_clip["effective_subtitle_text"] == "Hello"


def test_update_clip_bounds_rejects_absolute_range_past_part_end(tmp_path):
    manifest, projects_root, jobs_dir = _create_project(tmp_path)
    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    clip_id = manifest.clips[0].clip_id

    with pytest.raises(ValueError, match='past the end of its source part'):
        service.update_clip_bounds(manifest.project_id, clip_id, "00:01:20", "00:01:40")
