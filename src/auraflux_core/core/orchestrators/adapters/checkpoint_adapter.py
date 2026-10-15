import asyncio
import os
import re
import tempfile
from typing import Optional

from auraflux_core.core.orchestrators.ports import CheckpointPort

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class FileSystemCheckpointAdapter(CheckpointPort):
    """Filesystem-backed CheckpointPort with atomic, durable writes."""

    def __init__(self, storage_dir: str = "./checkpoints") -> None:
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_file_path(self, session_id: str) -> str:
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(f"Invalid session_id: {session_id!r}")
        return os.path.join(self.storage_dir, f"{session_id}.json")

    def _save_sync(self, file_path: str, state_json: str) -> None:
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=self.storage_dir, prefix=".ckpt-", suffix=".tmp",
                delete=False, encoding="utf-8",
            ) as tmp:
                tmp_path = tmp.name
                tmp.write(state_json)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, file_path)
            tmp_path = None  # replaced successfully; nothing to clean up
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _load_sync(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None

    async def save(self, session_id: str, state_json: str) -> None:
        file_path = self._get_file_path(session_id)
        try:
            await asyncio.to_thread(self._save_sync, file_path, state_json)
        except OSError as e:
            raise RuntimeError(f"Failed to save checkpoint for session {session_id}: {e}") from e

    async def load(self, session_id: str) -> Optional[str]:
        file_path = self._get_file_path(session_id)
        try:
            return await asyncio.to_thread(self._load_sync, file_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load checkpoint for session {session_id}: {e}") from e
