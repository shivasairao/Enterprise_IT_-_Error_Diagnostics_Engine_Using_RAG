import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# EnsembleRetriever moved to the `langchain-classic` package in LangChain 1.x
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # LangChain < 1.0
    from langchain.retrievers import EnsembleRetriever

# Model names change/retire often -- override via env vars without touching code.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gemini-3.5-flash")


def _ensure_event_loop():
    """Streamlit runs scripts in a worker thread with no asyncio loop; the Google
    clients need one or they raise 'There is no current event loop'."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _load_docs(data_path: Path) -> list[Document]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file '{data_path}' not found. Please create it first.")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [
        Document(
            page_content=(
                f"Error Code: {item['error_code']}\n"
                f"Symptom: {item['symptom']}\n"
                f"Resolution: {item['resolution']}"
            ),
            metadata={"error_code": item["error_code"]},
        )
        for item in raw_data
    ]


def get_hybrid_retriever_and_chain(
    api_key: str,
    data_file: str = "it_errors.json",
    faiss_dir: str = "faiss_index",
):
    """Returns (hybrid_retriever, rag_chain).
    rag_chain.invoke(query) -> {"docs": [...], "question": str, "answer": str}"""
    _ensure_event_loop()

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)

    data_path = Path(data_file)
    faiss_path = Path(faiss_dir)
    marker = faiss_path / "embedding_model.txt"
    docs = _load_docs(data_path)

    # 1. Dense index (FAISS): build once, rebuild if the embedding model changed
    index_ok = (
        (faiss_path / "index.faiss").exists()
        and marker.exists()
        and marker.read_text().strip() == EMBEDDING_MODEL
    )
    if not index_ok:
        faiss_vectorstore = FAISS.from_documents(docs, embeddings)
        faiss_vectorstore.save_local(str(faiss_path))
        marker.write_text(EMBEDDING_MODEL)
    else:
        faiss_vectorstore = FAISS.load_local(
            str(faiss_path), embeddings, allow_dangerous_deserialization=True
        )
    dense_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. Sparse index (BM25): cheap, rebuilt from the JSON every time (no pickle)
    bm25_retriever = BM25Retriever.from_documents(docs, k=2)

    # 3. Hybrid fusion (weighted Reciprocal Rank Fusion)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5],
    )

    # 4. Generation: retry on transient errors (e.g. 503), then fall back to a backup model
    primary = ChatGoogleGenerativeAI(
        model=LLM_MODEL, temperature=0, google_api_key=api_key, max_retries=1
    )
    backup = ChatGoogleGenerativeAI(
        model=FALLBACK_LLM_MODEL, temperature=0, google_api_key=api_key, max_retries=1
    )
    llm = primary.with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    ).with_fallbacks([backup])

    prompt = PromptTemplate.from_template(
        "You are an Enterprise IT Diagnostics Copilot. Diagnose the user's issue using ONLY the retrieved context below.\n"
        "If the context does not contain a relevant error, say so instead of guessing.\n\n"
        "Context:\n{context}\n\n"
        "User Issue: {question}\n\n"
        "Provide the exact Error Code and the step-by-step Resolution."
    )

    def format_docs(retrieved):
        return "\n\n".join(f"--- Document ---\n{d.page_content}" for d in retrieved)

    answer_chain = (
        RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )

    # Retrieve ONCE, then reuse the same docs for both the answer and the UI.
    rag_chain = (
        RunnableParallel(docs=hybrid_retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(answer=answer_chain)
    )

    return hybrid_retriever, rag_chain


if __name__ == "__main__":
    # Terminal test:  set GOOGLE_API_KEY=...  then  python RAG.py "dashboard is spinning"
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GOOGLE_API_KEY (or GEMINI_API_KEY) first.")
    _, chain = get_hybrid_retriever_and_chain(key)
    result = chain.invoke(sys.argv[1] if len(sys.argv) > 1 else "dashboard is spinning")
    print(result["answer"])