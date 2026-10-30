from pathlib import Path
from types import SimpleNamespace

import pytest

from core.editor import runtime


def test_ensure_editor_service_reuses_healthy_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime._ensure_runtime_dir()
    runtime._save_runtime_record({'pid': 123, 'port': 8765, 'host': '127.0.0.1', 'projects_root': str((tmp_path / 'processed_videos').resolve()), 'jobs_dir': str((tmp_path / 'jobs').resolve())})
    monkeypatch.setattr(runtime, '_is_process_alive', lambda pid: True)
    monkeypatch.setattr(runtime, '_healthy', lambda host, port, timeout=0.5: True)
    url = runtime.ensure_editor_service('proj-1', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs', open_browser=False)
    assert url == 'http://127.0.0.1:8765/projects/proj-1'


def test_ensure_editor_service_launches_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime._ensure_runtime_dir()
    monkeypatch.setattr(runtime, '_is_process_alive', lambda pid: False)
    monkeypatch.setattr(runtime, '_pick_free_port', lambda host='127.0.0.1': 9001)
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/usr/local/bin/uv' if name == 'uv' else None)

    calls = {'healthy': 0, 'opened': [], 'cmd': None}

    def fake_healthy(host, port, timeout=0.5):
        calls['healthy'] += 1
        return calls['healthy'] >= 2

    monkeypatch.setattr(runtime, '_healthy', fake_healthy)
    monkeypatch.setattr(runtime.webbrowser, 'open_new_tab', lambda url: calls['opened'].append(url))
    monkeypatch.setattr(
        runtime.subprocess,
        'Popen',
        lambda cmd, **kwargs: calls.update({'cmd': cmd}) or SimpleNamespace(pid=456, poll=lambda: None),
    )

    url = runtime.ensure_editor_service('proj-2', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs', open_browser=True)
    assert url == 'http://127.0.0.1:9001/projects/proj-2'
    assert calls['opened'] == [url]
    record = runtime._load_runtime_record()
    assert record['pid'] == 456
    assert record['port'] == 9001


def test_ensure_editor_service_uses_configured_public_base_url_for_reused_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENCLIP_EDITOR_BASE_URL', 'http://192.168.1.10:8765/')
    runtime._ensure_runtime_dir()
    runtime._save_runtime_record({'pid': 123, 'port': 8765, 'host': '0.0.0.0', 'projects_root': str((tmp_path / 'processed_videos').resolve()), 'jobs_dir': str((tmp_path / 'jobs').resolve())})
    monkeypatch.setattr(runtime, '_is_process_alive', lambda pid: True)
    monkeypatch.setattr(runtime, '_healthy', lambda host, port, timeout=0.5: host == '127.0.0.1' and port == 8765)

    url = runtime.ensure_editor_service('proj-lan', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs')

    assert url == 'http://192.168.1.10:8765/projects/proj-lan'


def test_ensure_editor_service_reuses_healthy_runtime_with_stale_pid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENCLIP_EDITOR_BASE_URL', 'http://192.168.1.10:8765/')
    runtime._ensure_runtime_dir()
    runtime._save_runtime_record({'pid': 123, 'port': 8765, 'host': '0.0.0.0', 'projects_root': str((tmp_path / 'processed_videos').resolve()), 'jobs_dir': str((tmp_path / 'jobs').resolve())})
    monkeypatch.setattr(runtime, '_is_process_alive', lambda pid: False)
    monkeypatch.setattr(runtime, '_healthy', lambda host, port, timeout=0.5: host == '127.0.0.1' and port == 8765)
    monkeypatch.setattr(
        runtime.subprocess,
        'Popen',
        lambda *args, **kwargs: pytest.fail('healthy editor runtime should be reused'),
    )

    url = runtime.ensure_editor_service('proj-lan', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs')

    assert url == 'http://192.168.1.10:8765/projects/proj-lan'


def test_ensure_editor_service_launches_lan_runtime_from_base_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENCLIP_EDITOR_BASE_URL', 'http://192.168.1.10:8765')
    runtime._ensure_runtime_dir()
    monkeypatch.setattr(runtime, '_is_process_alive', lambda pid: False)
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/usr/local/bin/uv' if name == 'uv' else None)

    calls = {'cmd': None}
    monkeypatch.setattr(runtime, '_healthy', lambda host, port, timeout=0.5: host == '127.0.0.1' and port == 8765)
    monkeypatch.setattr(
        runtime.subprocess,
        'Popen',
        lambda cmd, **kwargs: calls.update({'cmd': cmd}) or SimpleNamespace(pid=456, poll=lambda: None),
    )

    url = runtime.ensure_editor_service('proj-lan', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs')

    assert url == 'http://192.168.1.10:8765/projects/proj-lan'
    assert calls['cmd'][calls['cmd'].index('--host') + 1] == '0.0.0.0'
    assert calls['cmd'][calls['cmd'].index('--port') + 1] == '8765'
    record = runtime._load_runtime_record()
    assert record['host'] == '0.0.0.0'
    assert record['port'] == 8765


def test_ensure_editor_service_env_host_and_port_override_base_url_port(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENCLIP_EDITOR_BASE_URL', 'http://192.168.1.10:8765')
    monkeypatch.setenv('OPENCLIP_EDITOR_HOST', '127.0.0.1')
    monkeypatch.setenv('OPENCLIP_EDITOR_PORT', '9002')
    runtime._ensure_runtime_dir()
    monkeypatch.setattr(runtime, '_is_process_alive', lambda pid: False)
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: None)

    calls = {'cmd': None}
    monkeypatch.setattr(runtime, '_healthy', lambda host, port, timeout=0.5: host == '127.0.0.1' and port == 9002)
    monkeypatch.setattr(
        runtime.subprocess,
        'Popen',
        lambda cmd, **kwargs: calls.update({'cmd': cmd}) or SimpleNamespace(pid=789, poll=lambda: None),
    )

    url = runtime.ensure_editor_service('proj-lan', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs')

    assert url == 'http://192.168.1.10:8765/projects/proj-lan'
    assert calls['cmd'][calls['cmd'].index('--host') + 1] == '127.0.0.1'
    assert calls['cmd'][calls['cmd'].index('--port') + 1] == '9002'


def test_ensure_editor_service_rejects_invalid_env_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENCLIP_EDITOR_BASE_URL', '192.168.1.10:8765')
    with pytest.raises(ValueError, match='OPENCLIP_EDITOR_BASE_URL'):
        runtime.ensure_editor_service('proj-1', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs')

    monkeypatch.setenv('OPENCLIP_EDITOR_BASE_URL', 'http://192.168.1.10:not-a-port')
    with pytest.raises(ValueError, match='OPENCLIP_EDITOR_BASE_URL'):
        runtime.ensure_editor_service('proj-1', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs')

    monkeypatch.setenv('OPENCLIP_EDITOR_BASE_URL', 'http://192.168.1.10:8765')
    monkeypatch.setenv('OPENCLIP_EDITOR_PORT', 'not-a-port')
    with pytest.raises(ValueError, match='OPENCLIP_EDITOR_PORT'):
        runtime.ensure_editor_service('proj-1', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs')


def test_ensure_editor_service_does_not_reuse_runtime_with_wrong_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENCLIP_EDITOR_BASE_URL', 'http://192.168.1.10:8765')
    runtime._ensure_runtime_dir()
    runtime._save_runtime_record({'pid': 123, 'port': 8765, 'host': '127.0.0.1', 'projects_root': str((tmp_path / 'processed_videos').resolve()), 'jobs_dir': str((tmp_path / 'jobs').resolve())})
    monkeypatch.setattr(runtime, '_is_process_alive', lambda pid: True)
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/usr/local/bin/uv' if name == 'uv' else None)

    calls = {'cmd': None}
    monkeypatch.setattr(runtime, '_healthy', lambda host, port, timeout=0.5: host == '127.0.0.1' and port == 8765)
    monkeypatch.setattr(
        runtime.subprocess,
        'Popen',
        lambda cmd, **kwargs: calls.update({'cmd': cmd}) or SimpleNamespace(pid=456, poll=lambda: None),
    )

    url = runtime.ensure_editor_service('proj-lan', projects_root=tmp_path / 'processed_videos', jobs_dir=tmp_path / 'jobs')

    assert url == 'http://192.168.1.10:8765/projects/proj-lan'
    assert calls['cmd'] is not None
    assert runtime._load_runtime_record()['host'] == '0.0.0.0'
