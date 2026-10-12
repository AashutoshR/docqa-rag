"""Streamlit chat UI. Run with: streamlit run app.py"""

import streamlit as st

from docqa.chain import NOT_FOUND, ask
from docqa.config import Settings
from docqa.ingest import SUPPORTED_SUFFIXES, ingest_bytes
from docqa.retriever import Source
from docqa.store import get_chat_model, get_index, get_plan_model, get_reranker, get_vector_store

st.set_page_config(page_title="DocQA", page_icon="📄")

settings = Settings.from_env()
if missing := settings.missing_keys():
    st.error(f"Add {' and '.join(missing)} to your .env file, then restart the app.")
    st.stop()


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def load_resources():
    index = get_index(settings)
    store = get_vector_store(settings, index)
    return index, store, get_chat_model(settings), get_plan_model(settings), get_reranker(settings)


def show_sources(sources: list[Source]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for number, source in enumerate(sources, start=1):
            st.markdown(f"**[{number}] {source.label}** · similarity {source.score:.2f}")
            st.caption(source.text)


index, store, llm, planner, reranker = load_resources()
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Documents")
    namespace = st.text_input(
        "Document set",
        value=settings.namespace,
        help="Each set is a separate Pinecone namespace. Questions only search the current set.",
    ).strip() or settings.namespace

    uploads = st.file_uploader(
        "Upload files",
        type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
        accept_multiple_files=True,
    )
    if st.button("Index documents", disabled=not uploads, type="primary"):
        for upload in uploads:
            with st.spinner(f"Indexing {upload.name}..."):
                result = ingest_bytes(
                    upload.name,
                    upload.getvalue(),
                    store=store,
                    index=index,
                    namespace=namespace,
                    settings=settings,
                )
            if result.status == "empty":
                st.warning(f"{result.name}: no extractable text (scanned PDF?)")
            elif result.status == "skipped":
                st.info(f"{result.name}: already indexed")
            else:
                st.success(f"{result.name}: {result.status}, {result.chunks} chunks")

    st.header("Retrieval")
    top_k = st.slider("Chunks to retrieve", 1, 10, settings.top_k)
    min_score = st.slider(
        "Minimum similarity",
        0.0,
        1.0,
        settings.min_score,
        0.05,
        help="Chunks below this score are ignored. If none pass, the app says it couldn't find an answer.",
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("📄 DocQA")
st.caption("Ask questions about your uploaded documents. Answers cite the passages they came from.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if question := st.chat_input("Ask a question about your documents"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = ask(
                question,
                llm=llm,
                planner=planner,
                reranker=reranker,
                store=store,
                namespace=namespace,
                settings=settings,
                history=history,
                top_k=top_k,
                min_score=min_score,
            )
        text = st.write_stream(answer.stream)
        # Retrieved chunks that the model judged irrelevant are not sources.
        sources = [] if text.strip() == NOT_FOUND else answer.sources
        if sources:
            show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})
