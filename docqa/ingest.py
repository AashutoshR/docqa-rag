"""Load documents, split them into chunks and upsert them into Pinecone.

Run from the project root:
    python -m docqa.ingest data/sample
    python -m docqa.ingest report.pdf --namespace finance
"""

import argparse
import hashlib
import io
import re
import time
import unicodedata
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.exceptions import NotFoundException
from pypdf import PdfReader

from docqa.config import Settings
from docqa.word import doc_text, docx_text

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md"}
# Bump when clean_text or chunk_pages changes what gets stored for a file.
PIPELINE_VERSION = "3"
BATCH_SIZE = 64
RETRY_WAITS = (5, 15, 30, 60)  # seconds, for rate-limit responses


@dataclass
class IngestResult:
    name: str
    status: str  # "added" | "replaced" | "skipped" | "empty"
    chunks: int = 0
    pages: int = 0
    excerpts: tuple[str, ...] = ()  # a few passages from across the file


def load_pages(name: str, data: bytes) -> list[Document]:
    """Turn a file into one Document per page (PDF) or one for the whole file."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {name}")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    elif suffix == ".docx":
        pages = [(1, docx_text(data))]
    elif suffix == ".doc":
        pages = [(1, doc_text(data))]
    else:
        pages = [(1, data.decode("utf-8", errors="replace"))]

    return [
        Document(page_content=clean_text(text), metadata={"source": name, "page": number})
        for number, text in pages
        if text.strip()
    ]


def clean_text(text: str) -> str:
    """Undo typesetting artefacts that PDF extraction leaves in the text.

    Ligatures such as the single character for "fi" become plain letters, and
    words hyphenated across a line break ("state-" / "ments") are rejoined, so
    both the embedding model and keyword matching see real words.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)\u2010\s+(?=[a-z])", "", text)  # typesetter's hyphen
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    return text.replace("\u2010", "-")


def chunk_pages(pages: list[Document], settings: Settings) -> list[Document]:
    """Split a file into overlapping chunks that may run across page breaks.

    Splitting page by page would cut a passage in two wherever the book turns a
    page. Instead the pages are joined, split as one text, and each chunk is
    labelled with the pages it starts and ends on.
    """
    if not pages:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    # A single newline, so the splitter treats a page break like any other line
    # break instead of as a paragraph boundary it must split on.
    separator = "\n"
    page_starts, position = [], 0
    for page in pages:
        page_starts.append(position)
        position += len(page.page_content) + len(separator)
    whole = Document(
        page_content=separator.join(page.page_content for page in pages),
        metadata={"source": pages[0].metadata["source"]},
    )

    def page_at(position: int) -> int:
        return pages[bisect_right(page_starts, position) - 1].metadata["page"]

    chunks = splitter.split_documents([whole])
    for chunk in chunks:
        start = chunk.metadata.pop("start_index")
        chunk.metadata["page"] = page_at(start)
        chunk.metadata["page_end"] = page_at(start + len(chunk.page_content) - 1)
    return chunks


def sample_excerpts(chunks: list[Document], count: int = 4) -> tuple[str, ...]:
    """Evenly spaced chunks from the body of a file, skipping its front matter."""
    if len(chunks) <= count:
        return tuple(chunk.page_content for chunk in chunks)
    step = len(chunks) / (count + 1)
    return tuple(chunks[int(step * (i + 1))].page_content for i in range(count))


def id_prefix(name: str, workspace: str = "") -> str:
    """Prefix shared by every chunk of a file, whatever its content.

    A workspace's chunks all start with its ID, so one user's files can be
    listed and deleted without touching anyone else's.
    """
    owner = f"{workspace}:" if workspace else ""
    return owner + hashlib.sha1(name.encode("utf-8")).hexdigest()[:12] + ":"


def chunk_ids(name: str, data: bytes, count: int, workspace: str = "") -> list[str]:
    """IDs are derived from file name and content, so re-ingesting is idempotent.

    They also depend on PIPELINE_VERSION, so files indexed by an older version
    of the cleaning and chunking code are replaced when uploaded again.
    """
    content = hashlib.sha256(PIPELINE_VERSION.encode() + data).hexdigest()[:12]
    return [f"{id_prefix(name, workspace)}{content}:{i}" for i in range(count)]


def _existing_ids(index, prefix: str, namespace: str) -> list[str]:
    try:
        return [i for page in index.list(prefix=prefix, namespace=namespace) for i in page]
    except NotFoundException:
        # The namespace doesn't exist until the first upsert.
        return []


def _is_rate_limit(error: Exception) -> bool:
    return getattr(error, "status", None) == 429 or "429" in str(error)


def _add_with_retry(store, **kwargs) -> None:
    for wait in RETRY_WAITS:
        try:
            return store.add_texts(**kwargs)
        except Exception as error:
            if not _is_rate_limit(error):
                raise
            time.sleep(wait)
    store.add_texts(**kwargs)


def _delete_ids(index, ids: list[str], namespace: str) -> None:
    for start in range(0, len(ids), 1000):
        index.delete(ids=ids[start : start + 1000], namespace=namespace)


def ingest_bytes(
    name: str,
    data: bytes,
    *,
    store,
    index,
    namespace: str,
    settings: Settings,
    kind: str = "document",
    workspace: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> IngestResult:
    """Index one file. Unchanged files are skipped; changed files are replaced.

    Chunks are uploaded in batches, and chunks already in the index are not sent
    again, so an interrupted upload of a large book resumes where it stopped.
    """
    pages = load_pages(name, data)
    chunks = chunk_pages(pages, settings)
    if not chunks:
        return IngestResult(name, "empty")

    ids = chunk_ids(name, data, len(chunks), workspace)
    existing = set(_existing_ids(index, id_prefix(name, workspace), namespace))
    tags = {"kind": kind, "workspace": workspace} if workspace else {"kind": kind}
    excerpts = sample_excerpts(chunks)
    if existing == set(ids):
        return IngestResult(name, "skipped", len(ids), len(pages), excerpts)

    todo = [i for i, chunk_id in enumerate(ids) if chunk_id not in existing]
    done = len(ids) - len(todo)
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start : start + BATCH_SIZE]
        _add_with_retry(
            store,
            texts=[chunks[i].page_content for i in batch],
            metadatas=[{**chunks[i].metadata, **tags} for i in batch],
            ids=[ids[i] for i in batch],
            namespace=namespace,
        )
        done += len(batch)
        if on_progress:
            on_progress(done, len(ids))

    stale = sorted(existing - set(ids))
    _delete_ids(index, stale, namespace)
    return IngestResult(name, "replaced" if stale else "added", len(ids), len(pages), excerpts)


def delete_document(name: str, *, index, namespace: str, workspace: str = "") -> int:
    """Remove every chunk of a file from the index. Returns how many were removed."""
    ids = _existing_ids(index, id_prefix(name, workspace), namespace)
    _delete_ids(index, ids, namespace)
    return len(ids)


def find_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)


def main() -> None:
    from docqa.store import get_index, get_vector_store

    parser = argparse.ArgumentParser(description="Index documents into Pinecone.")
    parser.add_argument("path", type=Path, help="A file or a folder of PDF, Word, TXT or Markdown files")
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    if missing := settings.missing_keys(chat=False):
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")
    files = find_files(args.path)
    if not files:
        raise SystemExit(f"No PDF, Word, TXT or Markdown files found at {args.path}")

    index = get_index(settings)
    store = get_vector_store(settings, index)
    namespace = args.namespace or settings.namespace
    for path in files:
        result = ingest_bytes(
            path.name,
            path.read_bytes(),
            store=store,
            index=index,
            namespace=namespace,
            settings=settings,
        )
        print(f"{result.status:>8}  {result.chunks:>4} chunks  {result.name}")


if __name__ == "__main__":
    main()
