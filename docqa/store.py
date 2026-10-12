"""Pinecone index and model clients."""

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from docqa.config import Settings


def get_index(settings: Settings):
    """Return the Pinecone index, creating it on first use."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if pc.has_index(settings.index_name):
        dimension = pc.describe_index(settings.index_name).dimension
        if dimension != settings.embedding_dim:
            raise ValueError(
                f"Pinecone index '{settings.index_name}' holds {dimension}-dimension "
                f"vectors but EMBEDDING_DIM is {settings.embedding_dim}. "
                "Set PINECONE_INDEX to a new name for this embedding model."
            )
    else:
        pc.create_index(
            name=settings.index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            ),
        )
    return pc.Index(settings.index_name)


def get_embeddings(settings: Settings):
    if settings.embedding_provider == "pinecone":
        return PineconeEmbeddings(
            model=settings.embedding_model, pinecone_api_key=settings.pinecone_api_key
        )
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")


def get_vector_store(settings: Settings, index=None) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=index if index is not None else get_index(settings),
        embedding=get_embeddings(settings),
    )


def get_reranker(settings: Settings):
    """Return a function scoring (query, texts) -> list of relevance scores, or None."""
    if not settings.reranker:
        return None
    provider, _, model = settings.reranker.partition(":")
    if provider != "pinecone":
        raise ValueError(f"Unknown RERANKER provider: {settings.reranker}")
    inference = Pinecone(api_key=settings.pinecone_api_key).inference

    def score(query: str, texts: list[str]) -> list[float]:
        result = inference.rerank(
            model=model,
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=False,
            parameters={"truncate": "END"},
        )
        scores = [0.0] * len(texts)
        for item in result.data:
            scores[item.index] = item.score
        return scores

    return score


# Free tiers limit tokens per minute; waiting and retrying beats failing.
MAX_RETRIES = 6


def get_chat_model(settings: Settings):
    return init_chat_model(settings.chat_model, temperature=0, max_retries=MAX_RETRIES)


def get_plan_model(settings: Settings):
    return init_chat_model(settings.plan_model, temperature=0, max_retries=MAX_RETRIES)
