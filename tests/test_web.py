"""Offline tests for uploads with progress, the library record and the web API."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import docqa.ingest as ingest_module
import server
from docqa.chain import suggest_questions
from docqa.ingest import chunk_ids, delete_document, ingest_bytes, sample_excerpts
from docqa.library import Library
from docqa.retriever import retrieve
from test_pipeline import SETTINGS, TEXT, FakeIndex, FakeLLM, FakeStore, hit


def test_ingest_uploads_in_batches_and_reports_progress(monkeypatch):
    monkeypatch.setattr(ingest_module, "BATCH_SIZE", 2)
    index, progress = FakeIndex(), []
    store = FakeStore(index)
    result = ingest_bytes(
        "book.txt",
        TEXT,
        store=store,
        index=index,
        namespace="ns",
        settings=SETTINGS,
        kind="book",
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert result.chunks > 2 and result.pages == 1
    assert store.batches == -(-result.chunks // 2)
    assert progress[-1] == (result.chunks, result.chunks)
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)
    assert store.last_metadata == {"source": "book.txt", "page": 1, "page_end": 1, "kind": "book"}


def test_ingest_resumes_an_interrupted_upload(monkeypatch):
    monkeypatch.setattr(ingest_module, "BATCH_SIZE", 100)
    index = FakeIndex()
    total = ingest_bytes(
        "book.txt", TEXT, store=FakeStore(index), index=index, namespace="ns", settings=SETTINGS
    ).chunks
    ids = chunk_ids("book.txt", TEXT, total)
    for lost in ids[2:]:
        del index.vectors["ns"][lost]

    sent = []
    store = FakeStore(index)
    add = store.add_texts
    store.add_texts = lambda **kwargs: (sent.extend(kwargs["ids"]), add(**kwargs))
    result = ingest_bytes(
        "book.txt", TEXT, store=store, index=index, namespace="ns", settings=SETTINGS
    )
    assert result.status == "added"
    assert sent == ids[2:]
    assert set(index.vectors["ns"]) == set(ids)


def test_ingest_retries_when_rate_limited(monkeypatch):
    monkeypatch.setattr(ingest_module, "RETRY_WAITS", (0, 0))
    index = FakeIndex()
    store = FakeStore(index)
    add, failures = store.add_texts, [RuntimeError("429 Too Many Requests")]

    def flaky(**kwargs):
        if failures:
            raise failures.pop()
        add(**kwargs)

    store.add_texts = flaky
    result = ingest_bytes(
        "book.txt", TEXT, store=store, index=index, namespace="ns", settings=SETTINGS
    )
    assert result.status == "added" and len(index.vectors["ns"]) == result.chunks


def test_delete_document_removes_only_that_file():
    index = FakeIndex()
    for name in ("a.txt", "b.txt"):
        ingest_bytes(name, TEXT, store=FakeStore(index), index=index, namespace="ns", settings=SETTINGS)
    removed = delete_document("a.txt", index=index, namespace="ns")
    assert removed > 0
    assert set(index.vectors["ns"]) == set(chunk_ids("b.txt", TEXT, removed))


def test_retrieve_can_be_limited_to_chosen_documents():
    store = FakeStore(FakeIndex(), [hit("text", 0.9)])
    retrieve("q", store=store, namespace="ns", settings=SETTINGS)
    assert store.last_filter is None
    retrieve("q", store=store, namespace="ns", settings=SETTINGS, documents=["a.pdf"])
    assert store.last_filter == {"source": {"$in": ["a.pdf"]}}


def test_library_records_documents_per_collection(tmp_path):
    library = Library(tmp_path / "nested" / "library.json")
    assert library.list("main") == []
    library.add("main", "a.pdf", "book", 300, 900)
    library.add("other", "b.pdf", "paper", 12, 40)
    library.add("main", "a.pdf", "article", 301, 901)  # re-upload replaces the entry
    assert [(d["name"], d["kind"], d["pages"]) for d in library.list("main")] == [
        ("a.pdf", "article", 301)
    ]
    library.remove("main", "a.pdf")
    library.remove("main", "missing.pdf")
    assert library.list("main") == []
    assert len(Library(library.path).list("other")) == 1


class InlineWorker:
    def submit(self, function, *args):
        function(*args)


@pytest.fixture
def client(monkeypatch, tmp_path):
    index = FakeIndex()
    store = FakeStore(index, [hit("Leave is 24 days.", 0.9, source="handbook.txt", page=1)])
    llm = FakeLLM()
    bundle = SimpleNamespace(index=index, store=store, llm=llm, planner=llm, reranker=None)
    monkeypatch.setattr(server, "resources", lambda: bundle)
    monkeypatch.setattr(server, "library", Library(tmp_path / "library.json"))
    monkeypatch.setattr(server, "worker", InlineWorker())
    monkeypatch.setattr(server, "jobs", {})
    monkeypatch.setattr(server, "settings", SETTINGS)
    test_client = TestClient(server.app)
    test_client.index, test_client.store = index, store
    return test_client


def upload(client, name="handbook.txt", data=TEXT, kind="book"):
    return client.post("/api/upload", files={"file": (name, data)}, data={"kind": kind})


def test_home_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Upload a book, article or research paper" in page.text
    for asset in ("style.css", "app.js"):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"


def test_upload_indexes_the_file_and_lists_it(client):
    job = upload(client, kind="paper").json()
    assert job["status"] == "done" and job["done"] == job["total"] > 0
    listing = client.get("/api/library").json()
    assert listing["jobs"] == []
    assert [(d["name"], d["kind"], d["chunks"]) for d in listing["documents"]] == [
        ("handbook.txt", "paper", job["total"])
    ]
    assert len(client.index.vectors[SETTINGS.namespace]) == job["total"]


def test_upload_rejects_unsupported_files_and_types(client):
    assert upload(client, name="photo.png").status_code == 400
    assert upload(client, kind="podcast").status_code == 400
    assert client.get("/api/library").json() == {"documents": [], "jobs": []}


def test_unreadable_upload_is_reported_and_can_be_dismissed(client):
    job = upload(client, name="blank.txt", data=b"  ").json()
    assert job["status"] == "error" and "No readable text" in job["message"]
    assert client.get("/api/library").json()["jobs"] == [job]
    client.delete(f"/api/jobs/{job['id']}")
    assert client.get("/api/library").json() == {"documents": [], "jobs": []}


def test_remove_document_deletes_vectors_and_listing(client):
    upload(client, name="My Book (v2).txt")
    assert client.delete("/api/library/My%20Book%20(v2).txt").json()["removed"] > 0
    assert client.get("/api/library").json()["documents"] == []
    assert client.index.vectors[SETTINGS.namespace] == {}


def ask(client, **body):
    response = client.post("/api/ask", json=body)
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines()]


def test_ask_streams_sources_then_tokens_then_done(client):
    events = ask(client, question="How many leave days?", documents=["handbook.txt"])
    assert [e["type"] for e in events][0] == "sources"
    assert events[0]["sources"][0] | {"score": 0} == {
        "text": "Leave is 24 days.",
        "source": "handbook.txt",
        "page": 1,
        "score": 0,
        "kind": "document",
        "rerank_score": None,
        "page_end": 1,
    }
    assert "".join(e["text"] for e in events if e["type"] == "token").strip() == "Leave is 24 days [1]."
    assert events[-1] == {"type": "done", "found": True}
    assert client.store.last_filter == {"source": {"$in": ["handbook.txt"]}}


def test_ask_reports_not_found_and_uses_history(client):
    client.store.hits = [hit("unrelated", 0.1)]
    history = [{"role": "user", "content": "Tell me about leave"}]
    events = ask(client, question="How many days?", history=history)
    assert events[0] == {
        "type": "sources",
        "question": "How many days of annual leave?",
        "searches": ["annual leave days"],
        "sources": [],
    }
    assert events[-1] == {"type": "done", "found": False}


def test_ask_reports_errors_in_the_stream(client, monkeypatch):
    def broken():
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(server, "resources", broken)
    assert ask(client, question="q") == [
        {"type": "error", "message": "The free model is rate-limited right now. Wait a few seconds and try again."}
    ]


def test_upload_stores_suggested_questions_and_survives_a_failing_suggester(client, monkeypatch):
    upload(client)
    assert client.get("/api/library").json()["documents"][0]["questions"] == ["annual leave days"]

    def broken(llm, excerpts):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(server, "suggest_questions", broken)
    assert upload(client, name="other.txt").json()["status"] == "done"
    documents = {d["name"]: d for d in client.get("/api/library").json()["documents"]}
    assert documents["other.txt"]["questions"] == []


def test_suggest_questions_parses_a_json_list_and_tolerates_junk():
    good = FakeLLM(plan='Here you go: ["What is X?", " How does Y work? ", "", 7, "Third?", "Fourth?"]')
    assert suggest_questions(good, ["excerpt"]) == ["What is X?", "How does Y work?", "Third?"]
    assert suggest_questions(FakeLLM(plan="no list here"), ["excerpt"]) == []
    assert suggest_questions(good, []) == []


def test_sample_excerpts_are_spread_across_the_file():
    chunks = [SimpleNamespace(page_content=str(i)) for i in range(100)]
    assert sample_excerpts(chunks) == ("20", "40", "60", "80")
    assert sample_excerpts(chunks[:3]) == ("0", "1", "2")


def test_streamlit_app_answers_with_sources(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import docqa.store

    index = FakeIndex()
    store = FakeStore(index, [hit("Leave is 24 days.", 0.9)])
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("PINECONE_API_KEY", "test")
    monkeypatch.setattr(docqa.store, "get_index", lambda settings: index)
    monkeypatch.setattr(docqa.store, "get_vector_store", lambda settings, index=None: store)
    monkeypatch.setattr(docqa.store, "get_chat_model", lambda settings: FakeLLM())
    monkeypatch.setattr(docqa.store, "get_plan_model", lambda settings: FakeLLM())
    monkeypatch.setattr(docqa.store, "get_reranker", lambda settings: None)

    app = AppTest.from_file(str(server.WEB.parent / "app.py")).run(timeout=30)
    app.chat_input[0].set_value("How many leave days?").run(timeout=30)
    assert not app.exception
    reply = app.chat_message[-1]
    assert reply.markdown[0].value.strip() == "Leave is 24 days [1]."
    assert [expander.label for expander in reply.expander] == ["Sources (1)"]
