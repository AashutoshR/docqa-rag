"""A local record of which documents are indexed, so the UI can list them.

Pinecone can't list the distinct files behind its vectors, so the names, types
and sizes are kept in a small JSON file next to the project.
"""

import json
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

LIBRARY_FILE = Path(__file__).resolve().parent.parent / ".docqa" / "library.json"
KINDS = ("book", "article", "paper")


class Library:
    def __init__(self, path: Path = LIBRARY_FILE):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list(self, collection: str) -> list[dict]:
        """Documents in a collection, most recently added first."""
        with self._lock:
            documents = self._read().get(collection, {})
        return sorted(documents.values(), key=lambda d: d["added"], reverse=True)

    def add(
        self,
        collection: str,
        name: str,
        kind: str,
        pages: int,
        chunks: int,
        questions: Sequence[str] = (),
    ) -> None:
        with self._lock:
            data = self._read()
            data.setdefault(collection, {})[name] = {
                "name": name,
                "kind": kind,
                "pages": pages,
                "chunks": chunks,
                "questions": [*questions],
                "added": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._write(data)

    def remove(self, collection: str, name: str) -> None:
        with self._lock:
            data = self._read()
            if data.get(collection, {}).pop(name, None) is not None:
                self._write(data)
