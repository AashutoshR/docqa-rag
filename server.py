"""Web UI and JSON API. Run with: uvicorn server:app --port 8000"""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docqa.chain import NOT_FOUND, ask, suggest_questions
from docqa.config import Settings
from docqa.ingest import SUPPORTED_SUFFIXES, delete_document, ingest_bytes
from docqa.library import KINDS, Library
from docqa.store import get_chat_model, get_index, get_plan_model, get_reranker, get_vector_store

WEB = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

settings = Settings.from_env()
library = Library()
collection = f"{settings.index_name}/{settings.namespace}"

# Uploads are indexed one at a time to stay inside free-tier rate limits.
worker = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}

class FreshStaticFiles(StaticFiles):
    """Static files the browser re-checks on every load, so an updated page shows at once."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="DocQA")
app.mount("/static", FreshStaticFiles(directory=WEB), name="static")


@lru_cache
def resources():
    index = get_index(settings)
    return SimpleNamespace(
        index=index,
        store=get_vector_store(settings, index),
        llm=get_chat_model(settings),
        planner=get_plan_model(settings),
        reranker=get_reranker(settings),
    )


def describe(error: Exception) -> str:
    if getattr(error, "status_code", None) == 429 or "429" in str(error):
        return "The free model is rate-limited right now. Wait a few seconds and try again."
    return f"{type(error).__name__}: {error}"


def suggest(excerpts) -> list[str]:
    """Example questions for a new document. A nicety, so it never fails an upload."""
    try:
        return suggest_questions(resources().planner, list(excerpts))
    except Exception:
        return []


def run_job(job: dict, data: bytes) -> None:
    def on_progress(done: int, total: int) -> None:
        job.update(status="indexing", done=done, total=total)

    try:
        job.update(status="reading")
        result = ingest_bytes(
            job["name"],
            data,
            store=resources().store,
            index=resources().index,
            namespace=settings.namespace,
            settings=settings,
            kind=job["kind"],
            on_progress=on_progress,
        )
        if result.status == "empty":
            job.update(status="error", message="No readable text found. Is this a scanned PDF?")
            return
        library.add(
            collection,
            result.name,
            job["kind"],
            result.pages,
            result.chunks,
            suggest(result.excerpts),
        )
        job.update(status="done")
    except Exception as error:
        job.update(status="error", message=describe(error))


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status():
    return {
        "missing_keys": settings.missing_keys(),
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
    }


@app.get("/api/library")
def get_library():
    unfinished = [job for job in jobs.values() if job["status"] != "done"]
    return {"documents": library.list(collection), "jobs": unfinished}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), kind: str = Form("book")):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"{name or 'That file'} isn't a PDF, Word, TXT or Markdown file.")
    if kind not in KINDS:
        raise HTTPException(400, f"Unknown document type: {kind}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{name} is larger than 100 MB.")

    job = {"id": uuid.uuid4().hex, "name": name, "kind": kind, "status": "queued", "done": 0, "total": 0}
    jobs[job["id"]] = job
    worker.submit(run_job, job, data)
    return job


@app.delete("/api/jobs/{job_id}")
def dismiss_job(job_id: str):
    job = jobs.get(job_id)
    if job and job["status"] in ("done", "error"):
        del jobs[job_id]
    return {"ok": True}


@app.delete("/api/library/{name:path}")
def remove_document(name: str):
    removed = delete_document(name, index=resources().index, namespace=settings.namespace)
    library.remove(collection, name)
    return {"removed": removed}


class Message(BaseModel):
    role: str
    content: str


class Question(BaseModel):
    question: str
    history: list[Message] = []
    documents: list[str] | None = None


@app.post("/api/ask")
def ask_question(body: Question):
    """Stream the answer as newline-delimited JSON: sources, then tokens, then done."""

    def events():
        try:
            answer = ask(
                body.question,
                llm=resources().llm,
                planner=resources().planner,
                reranker=resources().reranker,
                store=resources().store,
                namespace=settings.namespace,
                settings=settings,
                history=[m.model_dump() for m in body.history],
                documents=body.documents,
            )
            yield {
                "type": "sources",
                "question": answer.question,
                "searches": answer.searches,
                "sources": [vars(s) for s in answer.sources],
            }
            text = ""
            for token in answer.stream:
                text += token
                yield {"type": "token", "text": token}
            yield {"type": "done", "found": answer.found and text.strip() != NOT_FOUND}
        except Exception as error:
            yield {"type": "error", "message": describe(error)}

    lines = (json.dumps(event) + "\n" for event in events())
    return StreamingResponse(lines, media_type="application/x-ndjson")
