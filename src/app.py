import os
import pickle
from pathlib import Path

import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from pipeline_manager import run_full_pipeline
from ticker_lookup import get_company_name

load_dotenv()

st.set_page_config(
    page_title="FinFaithful-RAG",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)


@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embed_model()


def load_index(ticker):
    index_path = Path(f"outputs/{ticker}_faiss.index")
    meta_path = Path(f"outputs/{ticker}_meta.pkl")

    if not index_path.exists() or not meta_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)

    return index, meta_data


def retrieve(ticker, query, k=3):
    index, meta_data = load_index(ticker)
    if index is None:
        return []

    q_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(meta_data["chunks"]):
            results.append(
                {
                    "text": meta_data["chunks"][idx],
                    "src": meta_data["meta"][idx]["source"],
                    "score": float(distances[0][i]),
                }
            )
    return results


def ask_rag(ticker, query):
    ctx = retrieve(ticker, query)
    if not ctx:
        return f"⚠️ No index found for {ticker}. Please run the pipeline first.", []

    context_str = "\n\n".join([f"Source: {c['src']}\n{c['text']}" for c in ctx])

    prompt = f"""You are a strict financial auditor AI analyzing SEC filings for {ticker}.
Rules:
1. Answer ONLY using the provided context.
2. If the answer is not present, say exactly: "Information not found in the provided filings."
3. Do not invent numbers or hallucinate.
4. Cite the source file names in your answer.

Context:
{context_str}

Question: {query}

Answer:"""

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a highly accurate financial AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            top_p=0.7,
            max_tokens=1024,
            stream=False,
        )
        return completion.choices[0].message.content, ctx
    except Exception as e:
        return f"⚠️ NVIDIA NIM API Error: {str(e)}", ctx


if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

with st.sidebar:
    st.title("🏦 FinFaithful-RAG")
    st.caption("Trustworthy Financial Document Q&A")
    st.divider()

    st.subheader("📊 Company Selection")
    ticker = st.text_input(
        "Enter Ticker Symbol",
        value="AAPL",
        help="e.g., AAPL, MSFT, TSLA, GOOGL",
    ).upper().strip()

    if ticker:
        try:
            company_name = get_company_name(ticker)
            st.success(f"**{company_name}** ({ticker})")
        except Exception:
            st.warning("Ticker not found in SEC database.")

    st.divider()

    if st.button("🚀 Download & Index Filings", type="primary", use_container_width=True):
        st.session_state.pipeline_status = []
        progress_placeholder = st.empty()

        def update_progress(msg):
            st.session_state.pipeline_status.append(msg)
            progress_placeholder.markdown("\n\n".join(st.session_state.pipeline_status))

        with st.spinner("Running pipeline..."):
            result = run_full_pipeline(ticker, progress_callback=update_progress)

        if result["status"] == "success":
            st.success(
                f"✅ Pipeline Complete!\n\n"
                f"- Filings Downloaded: {result['filings_downloaded']}\n"
                f"- Filings Parsed: {result['filings_parsed']}\n"
                f"- Chunks Indexed: {result['chunks_indexed']}"
            )
        else:
            st.error(f"❌ Pipeline Error: {result['message']}")

    st.divider()
    st.caption("Powered by NVIDIA NIM (Llama-3.1-70B) | SEC EDGAR | FAISS")

st.markdown('<p class="main-header">🏦 FinFaithful-RAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Trustworthy Financial Document Q&A — Powered by NVIDIA NIM & SEC EDGAR</p>',
    unsafe_allow_html=True,
)

if ticker not in st.session_state.chat_history:
    st.session_state.chat_history[ticker] = []

for message in st.session_state.chat_history[ticker]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved SEC Evidence"):
                for c in message["sources"]:
                    st.markdown(f"**Source:** `{c['src']}`")
                    st.text(c["text"][:800] + "...")

if prompt := st.chat_input(f"Ask a financial question about {ticker}'s SEC filings..."):
    st.session_state.chat_history[ticker].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing SEC filings with NVIDIA NIM..."):
            answer, context = ask_rag(ticker, prompt)
            st.markdown(answer)

            if context:
                with st.expander("🔍 View Retrieved SEC Evidence"):
                    for c in context:
                        st.markdown(f"**Source:** `{c['src']}`")
                        st.text(c["text"][:800] + "...")

            st.session_state.chat_history[ticker].append(
                {"role": "assistant", "content": answer, "sources": context}
            )
