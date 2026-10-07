import streamlit as st
from RAG import get_hybrid_retriever_and_chain

st.set_page_config(page_title="Enterprise IT Copilot", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner=False)
def init_pipeline(api_key: str):
    """Caches the heavy initialization of models and indices."""
    return get_hybrid_retriever_and_chain(api_key)


st.title("🛠️ Enterprise IT Diagnostics Engine")
st.caption("A Hybrid RAG fusing BM25 exact keyword matching with dense semantic vectors.")

# 1. Credential management (st.secrets raises if no secrets.toml exists)
api_key = None
try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    pass
if not api_key:
    api_key = st.sidebar.text_input("Google AI Studio API Key", type="password")

if not api_key:
    st.info("Please provide your Google AI API key in the sidebar to initialize the pipeline.")
    st.stop()

# 2. Pipeline initialization
with st.spinner("Initializing hybrid retrievers..."):
    try:
        retriever, rag_chain = init_pipeline(api_key)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.stop()

# 3. Chat interface
query = st.chat_input(
    "Enter an error code (e.g., 'HTTP_504') or describe the symptom (e.g., 'dashboard is spinning')."
)

if query:
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        with st.spinner("Diagnosing across sparse and dense indices..."):
            try:
                result = rag_chain.invoke(query)  # retrieves once
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()

            st.write(result["answer"])

            with st.expander("🔍 View Retrieved Hybrid Context (RRF Merged)"):
                for idx, doc in enumerate(result["docs"]):
                    st.markdown(f"**Result {idx + 1}:** `{doc.metadata.get('error_code', 'Unknown')}`")
                    st.text(doc.page_content)