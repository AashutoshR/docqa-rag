"""Settings, read once from the environment (and a local .env file)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

CHAT_KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    pinecone_api_key: str = ""

    index_name: str = "docqa-rag"
    namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # The defaults run entirely on free tiers (Groq + Pinecone's hosted embeddings).
    # "provider:model", so swapping to another provider is a config change,
    # e.g. CHAT_MODEL=openai:gpt-4o-mini.
    chat_model: str = "groq:openai/gpt-oss-120b"
    # A small, fast model that rewrites follow-ups and splits questions into
    # searches. On free tiers each model has its own rate limit, so this also
    # halves the load on the answer model.
    plan_model: str = "groq:openai/gpt-oss-20b"
    # "pinecone" for Pinecone's hosted models, or "openai", e.g. with
    # EMBEDDING_MODEL=text-embedding-3-small and EMBEDDING_DIM=1536.
    embedding_provider: str = "pinecone"
    embedding_model: str = "multilingual-e5-large"
    embedding_dim: int = 1024

    chunk_size: int = 800
    chunk_overlap: int = 120

    top_k: int = 5
    # "pinecone:<model>" re-scores retrieved chunks against the question with a
    # cross-encoder, which is slower but more accurate than vector similarity.
    # Empty turns reranking off.
    reranker: str = "pinecone:bge-reranker-v2-m3"
    # Chunks fetched per search for the reranker to choose the best top_k from.
    rerank_candidates: int = 20
    # Chunks scoring below this cosine similarity are dropped. If none are
    # left, the app says it couldn't find an answer instead of guessing.
    # The right value depends on the embedding model: multilingual-e5-large
    # scores almost everything between 0.7 and 0.9.
    min_score: float = 0.75

    # Hosting. With workspaces on, every browser gets its own private library.
    # Off, the default, keeps one shared library, which suits running locally.
    workspaces: bool = False
    # If set, the API only answers requests that send this code.
    access_code: str = ""
    max_upload_mb: int = 100
    # Documents allowed per library; 0 means no limit.
    max_documents: int = 0

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        d = cls()
        return cls(
            openai_api_key=env.get("OPENAI_API_KEY", ""),
            pinecone_api_key=env.get("PINECONE_API_KEY", ""),
            index_name=env.get("PINECONE_INDEX", d.index_name),
            namespace=env.get("PINECONE_NAMESPACE", d.namespace),
            pinecone_cloud=env.get("PINECONE_CLOUD", d.pinecone_cloud),
            pinecone_region=env.get("PINECONE_REGION", d.pinecone_region),
            chat_model=env.get("CHAT_MODEL", d.chat_model),
            plan_model=env.get("PLAN_MODEL", d.plan_model),
            embedding_provider=env.get("EMBEDDING_PROVIDER", d.embedding_provider).lower(),
            embedding_model=env.get("EMBEDDING_MODEL", d.embedding_model),
            embedding_dim=int(env.get("EMBEDDING_DIM", d.embedding_dim)),
            chunk_size=int(env.get("CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(env.get("CHUNK_OVERLAP", d.chunk_overlap)),
            top_k=int(env.get("TOP_K", d.top_k)),
            reranker=env.get("RERANKER", d.reranker),
            rerank_candidates=int(env.get("RERANK_CANDIDATES", d.rerank_candidates)),
            min_score=float(env.get("MIN_SCORE", d.min_score)),
            workspaces=env.get("WORKSPACES", "").strip().lower() in ("1", "on", "true", "yes"),
            access_code=env.get("ACCESS_CODE", "").strip(),
            max_upload_mb=int(env.get("MAX_UPLOAD_MB", d.max_upload_mb)),
            max_documents=int(env.get("MAX_DOCUMENTS", d.max_documents)),
        )

    def missing_keys(self, chat: bool = True) -> list[str]:
        """Names of the API keys this configuration needs but .env doesn't have."""
        needed = ["PINECONE_API_KEY"]
        if self.embedding_provider == "openai":
            needed.append("OPENAI_API_KEY")
        for model in (self.chat_model, self.plan_model) if chat else ():
            key = CHAT_KEY_NAMES.get(model.split(":")[0])
            if key and key not in needed:
                needed.append(key)
        return [name for name in needed if not os.environ.get(name)]
