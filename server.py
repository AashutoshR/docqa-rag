"""Web UI and JSON API. Run with: uvicorn server:app --port 8000"""

import hmac
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docqa.chain import NOT_FOUND, ask, suggest_questions
from docqa.config import Settings
from docqa.ingest import SUPPORTED_SUFFIXES, delete_document, ingest_bytes
from docqa.library import KINDS, Library
from docqa.store import get_chat_model, get_index, get_plan_model, get_reranker, get_vector_store

WEB = Path(__file__).parent / "web"
WORKSPACE_ID = re.compile(r"[a-z0-9]{16,64}")
ACTIVE = ("queued", "reading", "indexing")

log = logging.getLogger("docqa")
settings = Settings.from_env()

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
        library=Library(index, settings.namespace, settings.embedding_dim),
    )


def check_access(x_access_code: str = Header("")) -> None:
    """Reject the request unless it carries the access code, when one is set."""
    expected = settings.access_code
    if expected and not hmac.compare_digest(x_access_code.encode(), expected.encode()):
        raise HTTPException(401, "Enter the access code to use this app.")


def current_workspace(x_workspace: str = Header("")) -> str:
    """The caller's private library, or "" when the app runs with one shared library."""
    if not settings.workspaces:
        return ""
    if not WORKSPACE_ID.fullmatch(x_workspace):
        raise HTTPException(400, "This browser has no workspace ID. Reload the page.")
    return x_workspace


PROTECTED = [Depends(check_access)]


def describe(error: Exception) -> str:
    """A message safe to show a visitor; the detail goes to the server log."""
    if getattr(error, "status_code", None) == 429 or "429" in str(error):
        return "The free model is rate-limited right now. Wait a few seconds and try again."
    log.exception("Request failed", exc_info=error)
    return "Something went wrong on the server. Try again in a moment."


def public(job: dict) -> dict:
    return {key: value for key, value in job.items() if key != "workspace"}


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
            workspace=job["workspace"],
            on_progress=on_progress,
        )
        if result.status == "empty":
            job.update(status="error", message="No readable text found. Is this a scanned PDF?")
            return
        resources().library.add(
            job["workspace"],
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
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    """What the page needs before it can do anything else. Open to everyone."""
    return {
        "missing_keys": settings.missing_keys(),
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "access_code_required": bool(settings.access_code),
        "max_upload_mb": settings.max_upload_mb,
        "max_documents": settings.max_documents,
    }


@app.get("/api/library", dependencies=PROTECTED)
def get_library(workspace: str = Depends(current_workspace)):
    unfinished = [
        public(job)
        for job in jobs.values()
        if job["workspace"] == workspace and job["status"] != "done"
    ]
    return {"documents": resources().library.documents(workspace), "jobs": unfinished}


@app.post("/api/upload", dependencies=PROTECTED)
async def upload(
    file: UploadFile = File(...),
    kind: str = Form("book"),
    workspace: str = Depends(current_workspace),
):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"{name or 'That file'} isn't a PDF, Word, TXT or Markdown file.")
    if kind not in KINDS:
        raise HTTPException(400, f"Unknown document type: {kind}")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"{name} is larger than {settings.max_upload_mb} MB.")
    if settings.max_documents:
        names = {doc["name"] for doc in resources().library.documents(workspace)}
        names |= {
            job["name"]
            for job in jobs.values()
            if job["workspace"] == workspace and job["status"] in ACTIVE
        }
        if name not in names and len(names) >= settings.max_documents:
            raise HTTPException(
                409,
                f"This library is limited to {settings.max_documents} documents. "
                "Remove one before adding another.",
            )

    job = {
        "id": uuid.uuid4().hex,
        "name": name,
        "kind": kind,
        "status": "queued",
        "done": 0,
        "total": 0,
        "workspace": workspace,
    }
    jobs[job["id"]] = job
    worker.submit(run_job, job, data)
    return public(job)


@app.delete("/api/jobs/{job_id}", dependencies=PROTECTED)
def dismiss_job(job_id: str, workspace: str = Depends(current_workspace)):
    job = jobs.get(job_id)
    if job and job["workspace"] == workspace and job["status"] in ("done", "error"):
        del jobs[job_id]
    return {"ok": True}


@app.delete("/api/library/{name:path}", dependencies=PROTECTED)
def remove_document(name: str, workspace: str = Depends(current_workspace)):
    removed = delete_document(
        name, index=resources().index, namespace=settings.namespace, workspace=workspace
    )
    resources().library.remove(workspace, name)
    return {"removed": removed}


class Message(BaseModel):
    role: str
    content: str


class Question(BaseModel):
    question: str
    history: list[Message] = []
    documents: list[str] | None = None


@app.post("/api/ask", dependencies=PROTECTED)
def ask_question(body: Question, workspace: str = Depends(current_workspace)):
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
                workspace=workspace,
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
