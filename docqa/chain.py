"""Turn a question plus retrieved chunks into a cited answer."""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from docqa.config import Settings
from docqa.retriever import Source, retrieve_many

NOT_FOUND = "I couldn't find an answer to that in the uploaded documents."

ANSWER_SYSTEM = f"""You answer questions using only the numbered document excerpts provided.

Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge.
- After each claim, cite the excerpt it came from like [1] or [2][3]: the excerpt \
number in square brackets and nothing else.
- If the excerpts answer only part of the question, answer that part and say which \
part the documents don't cover.
- If the excerpts contain nothing that answers the question, reply with exactly: {NOT_FOUND}
- Be concise."""

PLAN_SYSTEM = """You prepare a question for searching a document library. Do not answer it.

Reply with JSON only: {"question": "...", "searches": ["...", "..."]}
- "question": the user's latest question rewritten to stand alone, resolving pronouns \
and references using the conversation. Leave it unchanged if it already stands alone.
- "searches": the search queries to run. Normally exactly one: the standalone question \
itself. Only when the question asks about two or three clearly different subjects (for \
example "What is X, and how does Y work?"), give one self-contained question per subject \
instead. Never add rephrasings, synonyms or variations of the same subject."""

SUGGEST_SYSTEM = """You write example questions for a document question-answering app. \
Given excerpts from one document, write {count} short, specific questions that the \
excerpts clearly answer, the kind a curious reader would ask about the subject matter. \
Each question must name its subject explicitly so that it makes sense with no other \
context: never write "the system", "the author", "the excerpt" or "the document". \
Reply with a JSON array of strings only."""

HISTORY_TURNS = 6
# Some models cite as [1†L3-L7]; only the source number is meaningful here.
CITATION_EXTRAS = re.compile(r"\[(\d+)†[^\]]*\]")
MAX_SEARCHES = 3

# Some models write citations as 【 1 】 and use typographic spaces and hyphens.
# Mapping them to plain characters keeps citations uniform and text searchable.
PLAIN_CHARS = str.maketrans(
    {
        "【": "[",
        "】": "]",
        " ": " ",  # narrow no-break space
        " ": " ",  # no-break space
        " ": " ",  # thin space
        "‑": "-",  # non-breaking hyphen
    }
)


@dataclass
class Answer:
    question: str  # the question rewritten to stand alone
    searches: list[str] = field(default_factory=list)  # the queries actually searched
    sources: list[Source] = field(default_factory=list)
    stream: Iterator[str] = field(default_factory=lambda: iter(()))

    @property
    def found(self) -> bool:
        return bool(self.sources)

    def text(self) -> str:
        return "".join(self.stream)


def plan_search(llm, history: list[dict], question: str) -> tuple[str, list[str]]:
    """Fold chat history into the question and split it into search queries.

    A follow-up such as "and for contractors?" becomes a full question, and a
    question about two topics gets one search per topic, so each is retrieved.
    Falls back to searching for the question as asked if the reply is unusable.
    """
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_TURNS:])
    reply = llm.invoke(
        [
            ("system", PLAN_SYSTEM),
            ("human", f"Conversation:\n{transcript or '(none)'}\n\nLatest question: {question}"),
        ]
    )
    try:
        plan = json.loads(re.search(r"\{.*\}", reply.text, re.DOTALL).group())
        standalone = plan["question"].strip()
        searches = [query.strip() for query in plan["searches"] if query.strip()]
    except (AttributeError, KeyError, TypeError, ValueError):
        return question, [question]
    standalone = standalone or question
    return standalone, searches[:MAX_SEARCHES] or [standalone]


def suggest_questions(llm, excerpts: list[str], count: int = 3) -> list[str]:
    """Example questions the excerpts can answer, for the UI to offer. Empty if unusable."""
    if not excerpts:
        return []
    reply = llm.invoke(
        [
            ("system", SUGGEST_SYSTEM.format(count=count)),
            ("human", "\n\n---\n\n".join(excerpts)),
        ]
    )
    try:
        questions = json.loads(re.search(r"\[.*\]", reply.text, re.DOTALL).group())
    except (AttributeError, ValueError):
        return []
    return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:count]


def format_context(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{i}] ({s.label})\n{s.text}" for i, s in enumerate(sources, start=1)
    )


def _plain_citations(tokens: Iterator[str]) -> Iterator[str]:
    """Rewrite citations such as [1†L3-L7] to [1] in a token stream.

    A citation can arrive split across tokens, so text after an unclosed "["
    is held back until the bracket closes.
    """
    pending = ""
    for token in tokens:
        pending += token
        opened = pending.rfind("[")
        if opened != -1 and "]" not in pending[opened:] and len(pending) - opened < 40:
            ready, pending = pending[:opened], pending[opened:]
        else:
            ready, pending = pending, ""
        if ready:
            yield CITATION_EXTRAS.sub(r"[\1]", ready)
    if pending:
        yield CITATION_EXTRAS.sub(r"[\1]", pending)


def _stream_answer(llm, question: str, sources: list[Source]) -> Iterator[str]:
    messages = [
        ("system", ANSWER_SYSTEM),
        ("human", f"Excerpts:\n\n{format_context(sources)}\n\nQuestion: {question}"),
    ]
    tokens = (chunk.text.translate(PLAIN_CHARS) for chunk in llm.stream(messages) if chunk.text)
    yield from _plain_citations(tokens)


def find_sources(
    question: str, *, llm, history: list[dict] | None = None, **search
) -> tuple[str, list[str], list[Source]]:
    """Plan the search and retrieve: (standalone question, searches, sources)."""
    standalone, searches = plan_search(llm, history or [], question)
    return standalone, searches, retrieve_many(searches, **search)


def ask(
    question: str,
    *,
    llm,
    store,
    namespace: str,
    settings: Settings,
    planner=None,
    reranker=None,
    history: list[dict] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    documents: list[str] | None = None,
) -> Answer:
    """Plan the search, retrieve, then answer.

    `planner` is the model used for the planning step (defaults to `llm`).
    With no relevant chunks the answer is a fixed refusal, not an LLM guess.
    """
    standalone, searches, sources = find_sources(
        question,
        llm=planner or llm,
        history=history,
        store=store,
        namespace=namespace,
        settings=settings,
        reranker=reranker,
        top_k=top_k,
        min_score=min_score,
        documents=documents,
    )
    if not sources:
        return Answer(standalone, searches, [], iter([NOT_FOUND]))
    return Answer(standalone, searches, sources, _stream_answer(llm, standalone, sources))
