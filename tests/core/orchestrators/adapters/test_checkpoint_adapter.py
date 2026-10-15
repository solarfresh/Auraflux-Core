import asyncio
import json
import os

import pytest

from auraflux_core.core.orchestrators.adapters.checkpoint_adapter import \
    FileSystemCheckpointAdapter


@pytest.fixture
def temp_storage(tmp_path):
    """Fixture providing a temporary directory for checkpoint storage."""
    return str(tmp_path / "checkpoints")


@pytest.mark.asyncio
async def test_save_and_load_success(temp_storage):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)
    session_id = "session-abc_123"
    state_json_v1 = '{"schema_version": "1.0.0", "session_id": "session-abc_123", "status": "IDLE", "data": "測試中文內容"}'
    state_json_v2 = '{"schema_version": "1.0.0", "session_id": "session-abc_123", "status": "PROCESSING", "data": "更新後內容"}'

    # 1. Test save & load with Unicode support
    await adapter.save(session_id, state_json_v1)
    assert await adapter.load(session_id) == state_json_v1

    # 2. Test overwrite (last-write-wins)
    await adapter.save(session_id, state_json_v2)
    assert await adapter.load(session_id) == state_json_v2


@pytest.mark.asyncio
async def test_load_nonexistent_session(temp_storage):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)
    result = await adapter.load("nonexistent_session")
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", [
    "a/b",          # Path traversal / collision risk
    "../..",        # Path traversal attempt
    "會話",          # Non-ASCII characters rejected by strict regex
    "abc\n",        # Trailing newline tested with fullmatch
    "a" * 129,      # Exceeds max length of 128
])
async def test_invalid_session_ids_rejected(temp_storage, invalid_id):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)

    with pytest.raises(ValueError):
        await adapter.save(invalid_id, "{}")

    with pytest.raises(ValueError):
        await adapter.load(invalid_id)


@pytest.mark.asyncio
async def test_empty_session_id_rejected(temp_storage):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)
    with pytest.raises(ValueError):
        await adapter.save("", "{}")

    # Ensure validation happens before any I/O and storage remains pristine
    assert os.listdir(temp_storage) == []


@pytest.mark.asyncio
async def test_session_id_length_boundary(temp_storage):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)
    valid_id_128 = "a" * 128

    # Exactly 128 chars should pass
    await adapter.save(valid_id_128, '{"status": "ok"}')
    assert await adapter.load(valid_id_128) == '{"status": "ok"}'


@pytest.mark.asyncio
async def test_concurrent_writes_same_session_never_corrupt(temp_storage):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)
    # Use large payloads to make race conditions or data tearing obvious if atomic writes fail
    payloads = [json.dumps({"id": i, "pad": "x" * 10_000}) for i in range(30)]

    # Concurrent writes targeting the EXACT same session ID
    await asyncio.gather(*(adapter.save("shared-session", p) for p in payloads))

    loaded = await adapter.load("shared-session")
    # Must equal one of the complete payloads (last-write-wins), never a mix or truncation
    assert loaded in payloads
    # No lingering temporary files
    assert [f for f in os.listdir(temp_storage) if f.endswith(".tmp")] == []


@pytest.mark.asyncio
async def test_temp_file_cleaned_up_when_replace_fails(temp_storage, monkeypatch):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)

    def boom(src, dst):
        raise OSError("disk exploded")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(RuntimeError, match="Failed to save checkpoint"):
        await adapter.save("session_test", '{"status": "OK"}')

    # Ensure no stray temp files or partial files left behind on failure
    assert os.listdir(temp_storage) == []


@pytest.mark.asyncio
async def test_failed_save_preserves_previous_checkpoint(temp_storage, monkeypatch):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)
    await adapter.save("s1", '{"v": 1}')

    # Simulate failure during atomic replacement
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("fail")))

    with pytest.raises(RuntimeError):
        await adapter.save("s1", '{"v": 2}')

    monkeypatch.undo()
    # Previous checkpoint must remain completely intact
    assert await adapter.load("s1") == '{"v": 1}'


@pytest.mark.asyncio
async def test_load_os_error_wrapped(temp_storage):
    adapter = FileSystemCheckpointAdapter(storage_dir=temp_storage)

    # Manufacture an OSError scenario: create a directory where a file is expected (IsADirectoryError)
    bad_path = os.path.join(temp_storage, "dir_as_file.json")
    os.makedirs(bad_path)

    with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
        await adapter.load("dir_as_file")
