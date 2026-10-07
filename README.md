# 🛠️ Enterprise IT & Error Diagnostics Engine (Hybrid RAG)

A Retrieval-Augmented Generation app that diagnoses IT errors from either an **exact error code** (`HTTP_504_GATEWAY_TIMEOUT`) or a **plain-English symptom** ("the dashboard keeps spinning") and returns the matching error code with a step-by-step resolution, grounded only in your knowledge base.

> Screenshot / demo GIF: `docs/demo.png` *(add yours here)*

## Why hybrid retrieval?

IT support queries come in two very different shapes:

| Query type | Example | Best handled by |
|---|---|---|
| Exact identifiers | `AUTH_TOKEN_EXPIRED_0x99` | **BM25** (sparse, keyword) |
| Natural-language symptoms | "users keep getting logged out" | **Dense vectors** (semantic) |

Pure vector search can miss exact codes; pure keyword search misses paraphrased symptoms. This project fuses both with **Reciprocal Rank Fusion (RRF)** so either style of query lands on the right runbook entry.

## Architecture

```
            ┌──────────────┐
 User query │  Streamlit   │
 ─────────► │  chat UI     │
            └──────┬───────┘
                   ▼
        ┌─────────────────────┐
        │  EnsembleRetriever  │   weights 0.5 / 0.5, RRF
        │   ┌──────┐ ┌──────┐ │
        │   │ BM25 │ │FAISS │ │   FAISS: Gemini embeddings
        │   └──────┘ └──────┘ │
        └──────────┬──────────┘
                   ▼  top-k fused docs (retrieved once)
        ┌─────────────────────┐
        │  Gemini LLM chain   │   retry w/ backoff → fallback model
        └──────────┬──────────┘
                   ▼
     Error code + resolution  (+ retrieved context shown in UI)
```

## Features

- **Hybrid search:** BM25 + FAISS dense retrieval fused with weighted RRF
- **Grounded answers:** the prompt restricts the LLM to retrieved context and tells it to say so when nothing relevant is found
- **Transparent:** an expander shows exactly which documents were retrieved
- **Resilient:** automatic retries with exponential backoff, then a fallback model on transient API errors (e.g. 503)
- **Efficient:** retrieval runs once per query; the same docs feed both the answer and the UI
- **Self-healing index:** the FAISS index rebuilds automatically if the embedding model changes
- **Configurable models:** swap models with environment variables, no code edits

## Tech stack

Python · Streamlit · LangChain (LCEL) · FAISS · rank_bm25 · Google Gemini (embeddings + chat)

## Project structure

```
├── app.py              # Streamlit UI
├── RAG.py              # Retrievers, fusion, LLM chain
├── it_errors.json      # Knowledge base (error_code, symptom, resolution)
├── requirements.txt
└── .streamlit/
    └── secrets.toml    # GEMINI_API_KEY (git-ignored)
```

## Getting started

```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free key from [Google AI Studio](https://aistudio.google.com/), then provide it in **one** of these ways:

1. `.streamlit/secrets.toml`
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
2. Environment variable `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
3. Paste it in the app's sidebar

Run:

```bash
streamlit run app.py
```

Terminal-only test:

```bash
export GOOGLE_API_KEY=your_key_here    # Windows: set GOOGLE_API_KEY=your_key_here
python RAG.py "users keep getting logged out"
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model for FAISS |
| `LLM_MODEL` | `gemini-3.8-flash` | Primary chat model |
| `FALLBACK_LLM_MODEL` | `gemini-3.5-flash` | Used if the primary keeps failing |

Gemini model names retire regularly. If you see a 404, the error message names the replacement; set it via the variables above.

## Adding your own knowledge base

Edit `it_errors.json`; each entry needs:

```json
{
  "error_code": "HTTP_504_GATEWAY_TIMEOUT",
  "symptom": "The frontend analytics dashboard spins indefinitely.",
  "resolution": "1. Increase ingress timeout...\n2. Add composite indexes..."
}
```

Delete the `faiss_index/` folder after changing the data so the dense index is rebuilt.

## Example queries

- `AUTH_TOKEN_EXPIRED_0x99`: exact-code lookup (BM25 shines)
- `dashboard is spinning`: symptom lookup (semantic search shines)
- `training job crashes on startup`: resolves to `CUDA_OUT_OF_MEMORY`

## Limitations & roadmap

- The sample dataset has only 3 entries; retrieval quality at scale is untested
- No evaluation harness yet (planned: hit-rate / MRR on a labeled query set, comparing BM25-only vs dense-only vs hybrid)
- Planned: cross-encoder reranking, chat memory, document ingestion from runbooks/PDFs, Docker deployment

## License

MIT *(add a LICENSE file)*
