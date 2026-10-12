"""Score the pipeline against a set of questions with known answers.

Run from the project root, after indexing the matching documents:
    python -m docqa.ingest data/sample --namespace sample
    python -m eval.run_eval --namespace sample
    python -m eval.run_eval --questions eval/books.jsonl --retrieval-only

Each line of the questions file is JSON:
    {"question": ..., "source": "file.pdf", "pages": [12, 13], "must_include": ["24"]}
"pages" is optional. A "must_include" entry may be a list of alternatives, any
one of which counts. A question that needs several documents lists them as
    {"question": ..., "evidence": [{"source": ..., "pages": [...]}, ...], "must_include": [...]}
and a question the documents cannot answer is
    {"question": ..., "answerable": false}
"""

import argparse
import json
import re
from pathlib import Path

from docqa.chain import NOT_FOUND, ask, find_sources
from docqa.config import Settings
from docqa.retriever import Source

DEFAULT_QUESTIONS = Path(__file__).parent / "questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evidence_of(case: dict) -> list[dict]:
    """The documents (and optionally pages) a correct answer has to draw on."""
    return case.get("evidence") or [{"source": case["source"], "pages": case.get("pages")}]


def supports(source: Source, evidence: dict) -> bool:
    pages = evidence.get("pages")
    return source.source == evidence["source"] and (
        not pages or any(page in pages for page in source.pages)
    )


def has_term(text: str, term: str | list[str]) -> bool:
    alternatives = [term] if isinstance(term, str) else term
    flat = " ".join(text.lower().split())  # a term may be broken across lines
    return any(alternative.lower() in flat for alternative in alternatives)


def score_retrieval(case: dict, sources: list[Source]) -> dict:
    evidence = evidence_of(case)
    found = [any(supports(s, item) for s in sources) for item in evidence]
    ranks = [i for i, s in enumerate(sources, start=1) if any(supports(s, item) for item in evidence)]
    return {
        # 1-based position of the first retrieved chunk from an expected page
        "rank": ranks[0] if ranks else None,
        "evidence_recall": sum(found) / len(found),
        # every expected term appears somewhere in the retrieved text
        "context_has_answer": all(
            any(has_term(s.text, term) for s in sources) for term in case["must_include"]
        ),
    }


def score_answer(case: dict, text: str) -> dict:
    refused = text.strip() == NOT_FOUND
    if not case.get("answerable", True):
        return {"refused": refused, "correct": refused}
    return {
        "refused": refused,
        "correct": all(has_term(text, term) for term in case["must_include"]),
        "cited": bool(re.search(r"\[\d+\]", text)),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-question rows into the headline metrics."""

    def rate(values: list) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "questions": len(rows),
        "retrieval_hit_rate": rate([r["evidence_recall"] == 1 for r in answerable]),
        "retrieval_mrr": rate([1 / r["rank"] if r["rank"] else 0 for r in answerable]),
        "context_recall": rate([r["context_has_answer"] for r in answerable]),
    }
    if any("correct" in r for r in rows):
        summary |= {
            "answer_accuracy": rate([r["correct"] for r in answerable]),
            "citation_rate": rate([r["cited"] for r in answerable]),
            "false_refusal_rate": rate([r["refused"] for r in answerable]),
            "correct_refusal_rate": rate([r["correct"] for r in unanswerable]),
        }
    return summary


def main() -> None:
    from docqa.store import get_chat_model, get_plan_model, get_reranker, get_vector_store

    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Plan and retrieve but don't generate answers; skips unanswerable questions",
    )
    parser.add_argument("--no-rerank", action="store_true", help="Turn the reranker off")
    parser.add_argument("--out", type=Path, default=None, help="Write full results as JSON")
    args = parser.parse_args()

    settings = Settings.from_env()
    if missing := settings.missing_keys():
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")
    planner = get_plan_model(settings)
    llm = None if args.retrieval_only else get_chat_model(settings)
    common = dict(
        store=get_vector_store(settings),
        namespace=args.namespace or settings.namespace,
        settings=settings,
        top_k=args.top_k,
        min_score=args.min_score,
        reranker=None if args.no_rerank else get_reranker(settings),
    )

    rows = []
    for case in load_questions(args.questions):
        answerable = case.get("answerable", True)
        if llm is None and not answerable:
            continue
        row = {"question": case["question"], "answerable": answerable}
        if llm is None:
            _, row["searches"], sources = find_sources(case["question"], llm=planner, **common)
        else:
            answer = ask(case["question"], llm=llm, planner=planner, **common)
            sources, row["searches"] = answer.sources, answer.searches
            row["answer"] = answer.text()
            row |= score_answer(case, row["answer"])
        row["retrieved"] = [s.label for s in sources]
        if answerable:
            row |= score_retrieval(case, sources)
        rows.append(row)

        good = row.get("correct", row.get("evidence_recall", 1) == 1 and row.get("context_has_answer", True))
        print(f"{'ok ' if good else 'BAD'} {case['question']}")

    summary = summarize(rows)
    print()
    for name, value in summary.items():
        print(f"{name:>22}: {value}")
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), "utf-8")


if __name__ == "__main__":
    main()
