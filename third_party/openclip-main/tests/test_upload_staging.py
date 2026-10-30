import io
from pathlib import Path

from core.upload_staging import (
    OWNER_SESSION_QUERY_PARAM,
    delete_upload_record,
    ensure_owner_session_id,
    list_uploads_for_owner,
    sanitize_uploaded_filename,
    stage_uploaded_file,
    uploads_root_for_output_dir,
)


class FakeUploadedFile(io.BytesIO):
    def __init__(self, name: str, content: bytes):
        super().__init__(content)
        self.name = name
        self.size = len(content)



def test_ensure_owner_session_id_sets_query_params_and_session_state():
    query_params = {}
    session_state = {}

    owner_session_id = ensure_owner_session_id(query_params, session_state)

    assert owner_session_id
    assert session_state[OWNER_SESSION_QUERY_PARAM] == owner_session_id
    assert query_params[OWNER_SESSION_QUERY_PARAM] == owner_session_id



def test_ensure_owner_session_id_prefers_existing_query_param():
    query_params = {OWNER_SESSION_QUERY_PARAM: "existing-token"}
    session_state = {}

    owner_session_id = ensure_owner_session_id(query_params, session_state)

    assert owner_session_id == "existing-token"
    assert session_state[OWNER_SESSION_QUERY_PARAM] == "existing-token"



def test_stage_uploaded_file_creates_owner_scoped_metadata(tmp_path):
    uploaded_file = FakeUploadedFile("my clip.MP4", b"video-bytes")

    metadata = stage_uploaded_file(uploaded_file, uploads_root_for_output_dir(tmp_path), "owner-a")

    assert metadata["owner_session_id"] == "owner-a"
    assert metadata["original_filename"] == "my clip.MP4"
    assert metadata["size_bytes"] == len(b"video-bytes")
    assert Path(metadata["staged_path"]).read_bytes() == b"video-bytes"
    uploads = list_uploads_for_owner(uploads_root_for_output_dir(tmp_path), "owner-a")
    assert len(uploads) == 1
    assert uploads[0]["upload_id"] == metadata["upload_id"]



def test_delete_upload_record_removes_upload_directory(tmp_path):
    uploaded_file = FakeUploadedFile("clip.mp4", b"video-bytes")
    metadata = stage_uploaded_file(uploaded_file, uploads_root_for_output_dir(tmp_path), "owner-a")

    delete_upload_record(metadata)

    assert not Path(metadata["staged_path"]).exists()
    assert list_uploads_for_owner(uploads_root_for_output_dir(tmp_path), "owner-a") == []



def test_sanitize_uploaded_filename_rejects_unsupported_extension():
    try:
        sanitize_uploaded_filename("notes.txt")
    except ValueError as exc:
        assert "Unsupported upload file type" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
