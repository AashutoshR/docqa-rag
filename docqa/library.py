"""The list of indexed documents, kept in Pinecone beside their vectors.

Pinecone can't list the distinct files behind its vectors, so each document
gets one small record in a side namespace holding its name, type and size.
Keeping these in Pinecone rather than in a local file means a hosted copy of
the app loses nothing when its server restarts.
"""

import hashlib
import json
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone

from pinecone.exceptions import NotFoundException

KINDS = ("book", "article", "paper")
FETCH_BATCH = 100
# Pinecone is eventually consistent, so a read straight after a write can miss
# it. Writes made by this process are replayed over reads for this long.
RECENT_SECONDS = 60


class Library:
    def __init__(self, index, namespace: str, dimension: int):
        self.index = index
        self.namespace = f"{namespace}-library"
        # Records need a vector, but these are never searched.
        self._placeholder = [1.0] + [0.0] * (dimension - 1)
        self._recent: dict[str, tuple[float, dict | None]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _prefix(workspace: str) -> str:
        return f"{workspace or 'shared'}:"

    def _id(self, workspace: str, name: str) -> str:
        return self._prefix(workspace) + hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]

    def _remember(self, record_id: str, entry: dict | None) -> None:
        with self._lock:
            self._recent[record_id] = (time.monotonic(), entry)

    def documents(self, workspace: str = "") -> list[dict]:
        """A workspace's documents, most recently added first."""
        prefix = self._prefix(workspace)
        try:
            pages = self.index.list(prefix=prefix, namespace=self.namespace)
            ids = [record_id for page in pages for record_id in page]
        except NotFoundException:
            ids = []  # the namespace doesn't exist until the first upload

        entries = {}
        for start in range(0, len(ids), FETCH_BATCH):
            batch = ids[start : start + FETCH_BATCH]
            fetched = self.index.fetch(ids=batch, namespace=self.namespace)
            for record_id, record in fetched.vectors.items():
                entries[record_id] = _from_metadata(record.metadata)

        now = time.monotonic()
        with self._lock:
            self._recent = {
                record_id: write
                for record_id, write in self._recent.items()
                if now - write[0] < RECENT_SECONDS
            }
            for record_id, (_, entry) in self._recent.items():
                if not record_id.startswith(prefix):
                    continue
                if entry is None:
                    entries.pop(record_id, None)
                else:
                    entries[record_id] = entry
        return sorted(entries.values(), key=lambda entry: entry["added"], reverse=True)

    def add(
        self,
        workspace: str,
        name: str,
        kind: str,
        pages: int,
        chunks: int,
        questions: Sequence[str] = (),
    ) -> None:
        entry = {
            "name": name,
            "kind": kind,
            "pages": pages,
            "chunks": chunks,
            "questions": [*questions],
            "added": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        record_id = self._id(workspace, name)
        metadata = {**entry, "questions": json.dumps(entry["questions"])}
        self.index.upsert(
            vectors=[{"id": record_id, "values": self._placeholder, "metadata": metadata}],
            namespace=self.namespace,
        )
        self._remember(record_id, entry)

    def remove(self, workspace: str, name: str) -> None:
        record_id = self._id(workspace, name)
        self.index.delete(ids=[record_id], namespace=self.namespace)
        self._remember(record_id, None)


def _from_metadata(metadata: dict) -> dict:
    """Pinecone returns numbers as floats and has no nested lists of its own."""
    return {
        "name": metadata["name"],
        "kind": metadata["kind"],
        "pages": int(metadata["pages"]),
        "chunks": int(metadata["chunks"]),
        "questions": json.loads(metadata.get("questions") or "[]"),
        "added": metadata["added"],
    }
