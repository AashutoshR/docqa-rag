"""Offline tests: Pinecone and the LLM are replaced with in-memory fakes."""

import io
import zipfile
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from docqa.chain import NOT_FOUND, ask, format_context
from docqa.config import Settings
from docqa.ingest import chunk_ids, chunk_pages, clean_text, ingest_bytes, load_pages
from docqa.retriever import Source, retrieve_many
from docqa.word import _tidy
from eval.run_eval import (
    DEFAULT_QUESTIONS,
    evidence_of,
    load_questions,
    score_answer,
    score_retrieval,
    summarize,
)

SETTINGS = Settings(chunk_size=200, chunk_overlap=20)
TEXT = ("Annual leave is 24 days. " * 30).encode()


class FakeIndex:
    """Chunk text and (for records written with upsert) metadata, by namespace and ID."""

    def __init__(self):
        self.vectors: dict[str, dict[str, str]] = {}
        self.records: dict[str, dict[str, dict]] = {}

    def list(self, prefix, namespace):
        stored = {**self.vectors.get(namespace, {}), **self.records.get(namespace, {})}
        ids = [i for i in stored if i.startswith(prefix)]
        if ids:
            yield ids

    def delete(self, ids, namespace):
        for i in ids:
            self.vectors.get(namespace, {}).pop(i, None)
            self.records.get(namespace, {}).pop(i, None)

    def upsert(self, vectors, namespace):
        for vector in vectors:
            self.records.setdefault(namespace, {})[vector["id"]] = vector["metadata"]

    def fetch(self, ids, namespace):
        stored = self.records.get(namespace, {})
        found = {i: SimpleNamespace(metadata=stored[i]) for i in ids if i in stored}
        return SimpleNamespace(vectors=found)


class FakeStore:
    def __init__(self, index, hits=()):
        self.index = index
        self.hits = list(hits)

    def add_texts(self, texts, metadatas, ids, namespace):
        self.batches = getattr(self, "batches", 0) + 1
        self.last_metadata = metadatas[-1]
        self.index.vectors.setdefault(namespace, {}).update(zip(ids, texts))

    def similarity_search_with_score(self, query, k, namespace, filter=None):
        self.queries = getattr(self, "queries", []) + [query]
        self.last_query = query
        self.last_filter = filter
        return self.hits[:k]


class FakeLLM:
    plan = '{"question": "How many days of annual leave?", "searches": ["annual leave days"]}'

    def __init__(self, reply="Leave is 24 days [1].", plan=plan):
        self.reply = reply
        self.plan = plan
        self.calls = 0
        self.answers = 0

    def invoke(self, messages):
        self.calls += 1
        self.last_plan_prompt = messages[-1][1]
        return SimpleNamespace(text=self.plan)

    def stream(self, messages):
        self.calls += 1
        self.answers += 1
        self.last_prompt = messages[-1][1]
        for word in self.reply.split(" "):
            yield SimpleNamespace(text=word + " ")


def minimal_pdf(text: str) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects[3] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    return bytes(out)


def ingest(name, data, index, namespace="ns"):
    return ingest_bytes(
        name, data, store=FakeStore(index), index=index, namespace=namespace, settings=SETTINGS
    )


def test_load_pdf_keeps_page_numbers():
    pages = load_pages("policy.pdf", minimal_pdf("Passwords need 14 characters"))
    assert [p.metadata for p in pages] == [{"source": "policy.pdf", "page": 1}]
    assert "14 characters" in pages[0].page_content


def test_load_reads_docx_paragraphs():
    body = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        "<w:p><w:r><w:t>Luxury brands rely on </w:t></w:r><w:r><w:t>exclusivity.</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", body)
    pages = load_pages("paper.docx", buffer.getvalue())
    assert pages[0].page_content == "Luxury brands rely on exclusivity.\nSecond paragraph."
    assert pages[0].metadata == {"source": "paper.docx", "page": 1}


def test_legacy_doc_text_keeps_field_results_and_drops_control_codes():
    raw = "See \x13 HYPERLINK \"http://x\" \x14the site\x15 now.\rCell\x07Next\x0cPage \x13 PAGE \x152\x01"
    assert _tidy(raw) == "See the site now.\nCell\tNext\nPage 2"


def test_load_rejects_unknown_file_type():
    with pytest.raises(ValueError):
        load_pages("photo.png", b"")


def test_chunks_respect_size_and_keep_metadata():
    chunks = chunk_pages(load_pages("handbook.md", TEXT), SETTINGS)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= SETTINGS.chunk_size for c in chunks)
    assert all(c.metadata == {"source": "handbook.md", "page": 1, "page_end": 1} for c in chunks)


def test_clean_text_repairs_ligatures_and_line_break_hyphens():
    ligature, soft_hyphen = chr(0xFB01), chr(0x2010)
    raw = f"The {ligature}rst suf{ligature}cient informa{soft_hyphen} tion state-\nments, self-contained Mini-\nMax"
    assert clean_text(raw) == "The first sufficient information statements, self-contained Mini-\nMax"


def test_chunks_run_across_page_breaks_and_keep_their_starting_page():
    def page(number, word):
        lines = "\n".join(f"{word} line {i}" for i in range(12))
        return Document(page_content=lines, metadata={"source": "b.pdf", "page": number})

    chunks = chunk_pages([page(3, "alpha"), page(4, "beta"), page(7, "gamma")], SETTINGS)
    assert [c.metadata["source"] for c in chunks] == ["b.pdf"] * len(chunks)
    assert sorted({c.metadata["page"] for c in chunks}) == [3, 4, 7]
    assert all(len(c.page_content) <= SETTINGS.chunk_size for c in chunks)
    crossing = [c for c in chunks if "alpha" in c.page_content and "beta" in c.page_content]
    assert crossing and (crossing[0].metadata["page"], crossing[0].metadata["page_end"]) == (3, 4)
    assert all(c.metadata["page"] <= c.metadata["page_end"] for c in chunks)
    assert chunk_pages([], SETTINGS) == []


def test_chunk_ids_depend_on_name_and_content():
    assert chunk_ids("a.md", b"x", 2) == chunk_ids("a.md", b"x", 2)
    assert chunk_ids("a.md", b"x", 1) != chunk_ids("a.md", b"y", 1)
    assert chunk_ids("a.md", b"x", 1) != chunk_ids("b.md", b"x", 1)


def test_ingest_adds_then_skips_unchanged_file():
    index = FakeIndex()
    first = ingest("handbook.md", TEXT, index)
    second = ingest("handbook.md", TEXT, index)
    assert (first.status, second.status) == ("added", "skipped")
    assert len(index.vectors["ns"]) == first.chunks


def test_ingest_replaces_changed_file_without_leftovers():
    index = FakeIndex()
    ingest("handbook.md", TEXT, index)
    ingest("other.md", TEXT, index)
    result = ingest("handbook.md", b"Annual leave is now 30 days.", index)
    assert result.status == "replaced"
    assert sorted(index.vectors["ns"].values()).count("Annual leave is now 30 days.") == 1
    assert set(chunk_ids("handbook.md", TEXT, 1)).isdisjoint(index.vectors["ns"])
    assert set(chunk_ids("other.md", TEXT, 1)) <= set(index.vectors["ns"])


def test_ingest_keeps_namespaces_separate():
    index = FakeIndex()
    ingest("handbook.md", TEXT, index, namespace="a")
    assert ingest("handbook.md", TEXT, index, namespace="b").status == "added"


def test_ingest_reports_empty_file():
    assert ingest("blank.txt", b"   \n", FakeIndex()).status == "empty"


def hit(text, score, source="handbook.md", page=3):
    return Document(page_content=text, metadata={"source": source, "page": page}), score


def run_ask(hits, llm, **kwargs):
    return ask(
        "How many leave days?",
        llm=llm,
        store=FakeStore(FakeIndex(), hits),
        namespace="ns",
        settings=SETTINGS,
        **kwargs,
    )


def test_ask_answers_from_sources_above_threshold():
    llm = FakeLLM()
    answer = run_ask([hit("weak", 0.1), hit("Leave is 24 days.", 0.8)], llm)
    assert [s.text for s in answer.sources] == ["Leave is 24 days."]
    assert answer.text().strip() == "Leave is 24 days [1]."
    assert "[1] (handbook.md, p. 3)\nLeave is 24 days." in llm.last_prompt
    assert "weak" not in llm.last_prompt


def test_answer_text_is_normalised_to_plain_citations_and_spaces():
    llm = FakeLLM("Within\u202f30\u202fdays for full\u2011time staff\u30101\u3011.")
    answer = run_ask([hit("Claims within 30 days.", 0.8)], llm)
    assert answer.text().strip() == "Within 30 days for full-time staff[1]."


def test_citations_in_other_styles_become_plain_numbers_even_when_split_across_tokens():
    dagger = chr(0x2020)
    reply = f"41 bits[1{dagger}L1-L4][4{dagger}L2-L5] and a[0] plus [2]. Open [bracket"
    llm = FakeLLM(reply)
    llm.stream = lambda messages: (SimpleNamespace(text=character) for character in reply)
    answer = run_ask([hit("x", 0.9)], llm)
    assert answer.text() == "41 bits[1][4] and a[0] plus [2]. Open [bracket"


def test_source_label_shows_a_page_range_when_the_chunk_crosses_pages():
    assert Source("t", "a.pdf", 16, 0.9).label == "a.pdf, p. 16"
    assert Source("t", "a.pdf", 16, 0.9, page_end=17).label == "a.pdf, pp. 16-17"
    assert list(Source("t", "a.pdf", 16, 0.9, page_end=17).pages) == [16, 17]


def test_ask_refuses_without_calling_llm_when_nothing_is_relevant():
    llm = FakeLLM()
    answer = run_ask([hit("unrelated", 0.05)], llm)
    assert not answer.found
    assert answer.text() == NOT_FOUND
    assert llm.answers == 0


def test_ask_rewrites_follow_up_questions_before_searching():
    llm = FakeLLM()
    store = FakeStore(FakeIndex(), [hit("Leave is 24 days.", 0.8)])
    history = [{"role": "user", "content": "Tell me about annual leave"}]
    answer = ask(
        "How many days?", llm=llm, store=store, namespace="ns", settings=SETTINGS, history=history
    )
    assert "user: Tell me about annual leave" in llm.last_plan_prompt
    assert answer.question == "How many days of annual leave?"
    assert store.queries == answer.searches == ["annual leave days"]


def test_ask_searches_each_topic_of_a_multi_part_question():
    searches = '["microservice size", "function size", "", "c", "d"]'
    plan = f'```json\n{{"question": "q", "searches": {searches}}}\n```'
    llm = FakeLLM(plan=plan)
    store = FakeStore(FakeIndex(), [hit("a", 0.9), hit("b", 0.8)])
    answer = ask("q", llm=llm, store=store, namespace="ns", settings=SETTINGS)
    assert store.queries == answer.searches == ["microservice size", "function size", "c"]
    assert [s.text for s in answer.sources] == ["a", "b"]  # duplicates across searches merged


def test_ask_falls_back_to_the_question_when_the_plan_is_unusable():
    for plan in ("Sure! Here you go.", '{"question": "x"}', '{"question": 5, "searches": []}'):
        store = FakeStore(FakeIndex(), [hit("a", 0.9)])
        answer = ask("How big?", llm=FakeLLM(plan=plan), store=store, namespace="ns", settings=SETTINGS)
        assert (answer.question, store.queries) == ("How big?", ["How big?"])


def test_retrieve_many_interleaves_results_and_caps_the_total():
    class TopicStore:
        def similarity_search_with_score(self, query, k, namespace):
            return [hit(f"{query}{i}", 0.9 - i / 100) for i in range(k)]

    merged = retrieve_many(["a", "b"], store=TopicStore(), namespace="ns", settings=SETTINGS, top_k=5)
    assert [s.text for s in merged] == ["a0", "b0", "a1", "b1", "a2", "b2", "a3", "b3"]
    single = retrieve_many(["a"], store=TopicStore(), namespace="ns", settings=SETTINGS, top_k=5)
    assert len(single) == 5


def test_format_context_numbers_sources_from_one():
    context = format_context([Source("a", "x.pdf", 2, 0.9), Source("b", "y.md", 1, 0.5)])
    assert context == "[1] (x.pdf, p. 2)\na\n\n[2] (y.md, p. 1)\nb"


def test_eval_question_files_are_well_formed():
    for path in (DEFAULT_QUESTIONS, DEFAULT_QUESTIONS.with_name("library.jsonl")):
        cases = load_questions(path)
        assert len(cases) >= 10
        for case in cases:
            assert case["question"]
            if case.get("answerable", True):
                assert case["must_include"]
                assert all(item["source"] for item in evidence_of(case))


def test_eval_scores_retrieval_by_page_and_expected_terms():
    case = {"question": "q", "source": "a.pdf", "pages": [5, 6], "must_include": ["24", ["leave", "holiday"]]}
    wrong_page = Source("24 days of holiday", "a.pdf", 9, 0.9)
    wrong_file = Source("x", "b.pdf", 5, 0.9)
    right = Source("y", "a.pdf", 6, 0.8)
    assert score_retrieval(case, [wrong_page, wrong_file, right]) == {
        "rank": 3,
        "evidence_recall": 1.0,
        "context_has_answer": True,
    }
    assert score_retrieval(case, [wrong_file]) == {
        "rank": None,
        "evidence_recall": 0.0,
        "context_has_answer": False,
    }
    without_pages = {"question": "q", "source": "a.pdf", "must_include": ["x"]}
    assert score_retrieval(without_pages, [wrong_page])["rank"] == 1

    two_books = {
        "question": "q",
        "evidence": [{"source": "a.pdf", "pages": [6]}, {"source": "c.pdf"}],
        "must_include": ["y"],
    }
    assert score_retrieval(two_books, [wrong_file, right])["evidence_recall"] == 0.5


def test_eval_scores_answers_and_summarises():
    case = {"question": "q", "source": "a.md", "must_include": [["24", "twenty-four"]]}
    good = score_answer(case, "It is Twenty-Four days [1].")
    assert good == {"refused": False, "correct": True, "cited": True}
    assert score_answer({"question": "q", "answerable": False}, NOT_FOUND)["correct"]

    found = {"rank": 2, "evidence_recall": 1.0, "context_has_answer": True}
    missed = {"rank": None, "evidence_recall": 0.5, "context_has_answer": False}
    rows = [
        {"answerable": True, **found, **good},
        {"answerable": True, **missed, **score_answer(case, NOT_FOUND)},
        {"answerable": False, "refused": True, "correct": True},
    ]
    assert summarize(rows) == {
        "questions": 3,
        "retrieval_hit_rate": 0.5,
        "retrieval_mrr": 0.25,
        "context_recall": 0.5,
        "answer_accuracy": 0.5,
        "citation_rate": 0.5,
        "false_refusal_rate": 0.5,
        "correct_refusal_rate": 1.0,
    }
    assert "answer_accuracy" not in summarize([{"answerable": True, **found}])


def test_reranker_reorders_candidates_and_keeps_top_k():
    class WideStore:
        def similarity_search_with_score(self, query, k, namespace):
            self.asked_for = k
            return [hit(f"chunk{i}", 0.9 - i / 100) for i in range(k)]

    def prefer_later(query, texts):
        return [float(text.removeprefix("chunk")) for text in texts]

    settings = Settings(top_k=3, rerank_candidates=10)
    store = WideStore()
    sources = retrieve_many(["q"], store=store, namespace="ns", settings=settings, reranker=prefer_later)
    assert store.asked_for == 10
    assert [s.text for s in sources] == ["chunk9", "chunk8", "chunk7"]
    assert sources[0].rerank_score == 9.0

    retrieve_many(["q"], store=store, namespace="ns", settings=settings)
    assert store.asked_for == 3


def test_reranker_failure_falls_back_to_vector_order():
    class Store:
        def similarity_search_with_score(self, query, k, namespace):
            return [hit(f"chunk{i}", 0.9 - i / 100) for i in range(k)]

    def out_of_quota(query, texts):
        raise RuntimeError("429 monthly limit reached")

    settings = Settings(top_k=2, rerank_candidates=6)
    sources = retrieve_many(["q"], store=Store(), namespace="ns", settings=settings, reranker=out_of_quota)
    assert [s.text for s in sources] == ["chunk0", "chunk1"]
