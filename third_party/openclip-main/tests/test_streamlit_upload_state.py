from core.browser_session import (
    INPUT_TYPE_SERVER_PATH,
    INPUT_TYPE_URL,
    normalize_input_type,
    reset_browser_state,
)


def test_normalize_input_type_migrates_legacy_local_file_value():
    assert normalize_input_type("Local File") == INPUT_TYPE_SERVER_PATH
    assert normalize_input_type("Video URL") == INPUT_TYPE_URL


def test_reset_browser_state_does_not_inherit_shared_persisted_runtime_values():
    defaults = {
        "input_type": INPUT_TYPE_URL,
        "video_source": "",
        "processing_result": None,
        "ui_language": "zh",
    }
    persisted = {
        "input_type": "Local File",
        "video_source": "/tmp/video.mp4",
        "processing_result": {"success": True},
        "ui_language": "en",
    }

    state = reset_browser_state(defaults)

    assert state == defaults
    assert defaults["input_type"] == INPUT_TYPE_URL
    assert defaults["processing_result"] is None


def test_reset_browser_state_returns_clean_copy():
    defaults = {
        "input_type": INPUT_TYPE_URL,
        "video_source": "",
        "processing_result": None,
    }

    reset = reset_browser_state(defaults)
    reset["video_source"] = "changed"

    assert defaults["video_source"] == ""
