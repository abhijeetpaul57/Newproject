from __future__ import annotations

import os
import requests
import streamlit as st

st.set_page_config(
    page_title="DocuFlow AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom design system CSS (modern typography, sleek dark palette, glassmorphism, glowing accents)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #0A0D14;
        --bg-card: rgba(17, 24, 39, 0.7);
        --bg-card-hover: rgba(26, 36, 59, 0.85);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-focus: rgba(99, 102, 241, 0.5);
        --primary-indigo: #6366F1;
        --primary-violet: #8B5CF6;
        --primary-cyan: #06B6D4;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --success-green: #10B981;
        --danger-rose: #F43F5E;
    }

    /* Base Body & Fonts */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.07) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 40%);
        background-attachment: fixed;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Custom Hero Header */
    .hero-container {
        padding: 1.25rem 0 1.5rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        font-size: 0.78rem;
        font-weight: 600;
        color: #A5B4FC;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--primary-cyan);
        box-shadow: 0 0 10px var(--primary-cyan);
        animation: pulseAnimation 2s infinite;
    }
    @keyframes pulseAnimation {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }
    .hero-title {
        font-size: 2.25rem;
        margin: 0;
        background: linear-gradient(135deg, #FFFFFF 20%, #CBD5E1 60%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c101d !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
        margin: 1.2rem 0 !important;
    }

    /* Sidebar Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.6) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 12px !important;
        padding: 0.85rem 1rem !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }

    /* Radio Navigation Tabs in Sidebar */
    div[data-testid="stRadio"] > div {
        gap: 0.5rem;
    }
    div[data-testid="stRadio"] label {
        background: rgba(23, 31, 51, 0.4) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        padding: 0.65rem 0.9rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.35) !important;
    }

    /* Buttons Styling */
    .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.15rem !important;
        border: 1px solid var(--border-subtle) !important;
        background: rgba(23, 31, 51, 0.7) !important;
        color: #E2E8F0 !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        border-color: var(--primary-indigo) !important;
        background: rgba(99, 102, 241, 0.18) !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.25) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Prompt Suggestion Chips */
    .stButton > button[key^="sample_"] {
        font-size: 0.85rem !important;
        text-align: left !important;
        border-radius: 10px !important;
        background: rgba(17, 24, 39, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 0.75rem 0.85rem !important;
        min-height: 4rem !important;
        display: flex !important;
        align-items: center !important;
    }
    .stButton > button[key^="sample_"]:hover {
        background: rgba(99, 102, 241, 0.12) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
    }

    /* Chat Messages */
    .stChatMessage {
        border-radius: 14px !important;
        padding: 1.15rem !important;
        margin-bottom: 0.9rem !important;
        border: 1px solid var(--border-subtle) !important;
        background: rgba(17, 24, 39, 0.55) !important;
        backdrop-filter: blur(14px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    }
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #06B6D4, #3B82F6) !important;
    }

    /* Chat Input Container */
    [data-testid="stChatInput"] {
        border-radius: 14px !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--primary-violet) !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3), 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    }

    /* Expander (Trace & Citations) */
    .streamlit-expanderHeader {
        background: rgba(23, 31, 51, 0.5) !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
        border: 1px solid var(--border-subtle) !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stExpander"] div[role="region"] {
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        background: rgba(13, 19, 33, 0.6) !important;
        padding: 1rem !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(17, 24, 39, 0.45) !important;
        border: 1.5px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-indigo) !important;
        background: rgba(99, 102, 241, 0.05) !important;
    }

    /* Text Inputs */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1px solid var(--border-subtle) !important;
        background: rgba(17, 24, 39, 0.6) !important;
        color: #F8FAFC !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-indigo) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Quick Document List Card in Sidebar */
    .doc-sidebar-badge {
        display: block;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        background: rgba(23, 31, 51, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.82rem;
    }
    .doc-sidebar-badge:hover {
        border-color: rgba(99, 102, 241, 0.3);
        background: rgba(23, 31, 51, 0.8);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def fetch_health():
    """Query FastAPI health endpoint to retrieve status and index statistics."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    if "127.0.0.1" in API_BASE_URL:
        try:
            res = requests.get(f"{API_BASE_URL.replace('127.0.0.1', 'localhost')}/health", timeout=3)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
    return None


def fetch_documents():
    """Fetch list of all currently indexed documents from the API."""
    try:
        res = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if res.status_code == 200:
            return res.json().get("documents", [])
    except Exception:
        pass
    return []


def clear_all_documents():
    """Send request to API to delete all indexed documents from ChromaDB."""
    try:
        res = requests.delete(f"{API_BASE_URL}/documents", timeout=10)
        return res.status_code == 200
    except Exception:
        return False


def delete_single_document(doc_id: str):
    """Delete a specific document and its chunks by doc_id via the API."""
    try:
        res = requests.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=10)
        return res.status_code == 200
    except Exception:
        return False


def submit_query(question: str, top_k: int = 5):
    """Send a question payload to the /query endpoint and return pipeline response."""
    res = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    try:
        data = res.json()
    except Exception:
        data = {"detail": res.text}
    if res.status_code == 200:
        return data
    else:
        err_msg = data.get("detail", res.text)
        raise RuntimeError(f"API Error ({res.status_code}): {err_msg}")


def submit_feedback(rating: str, comment: str, question: str):
    """Submit a thumbs up/down evaluation with optional comment to /feedback."""
    try:
        res = requests.post(
            f"{API_BASE_URL}/feedback",
            json={"rating": rating, "comment": comment, "question": question},
            timeout=10,
        )
        return res.status_code == 200
    except Exception:
        return False


def upload_files(files):
    """Transmit uploaded files multipart-form data to the /ingest endpoint."""
    upload_list = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
    res = requests.post(f"{API_BASE_URL}/ingest", files=upload_list, timeout=120)
    try:
        data = res.json()
    except Exception:
        data = {"detail": res.text}
    if res.status_code == 200:
        return data
    raise RuntimeError(f"Upload failed ({res.status_code}): {data.get('detail', res.text)}")


def ingest_url(url: str):
    """Send remote documentation URL payload to the /ingest/url endpoint."""
    res = requests.post(
        f"{API_BASE_URL}/ingest/url",
        json={"urls": [url]},
        timeout=30,
    )
    try:
        data = res.json()
    except Exception:
        data = {"detail": res.text}
    if res.status_code == 200:
        return data
    raise RuntimeError(f"URL ingest failed ({res.status_code}): {data.get('detail', res.text)}")


with st.sidebar:
    st.markdown("### ⚙️ System Control")
    health = fetch_health()

    if health:
        st.success(f"🟢 **API Online** (`{health.get('llm_model', 'N/A')}`)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", health.get("indexed_documents", 0))
        with col2:
            st.metric("Total Chunks", health.get("indexed_chunks", 0))
    else:
        st.error(f"🔴 **FastAPI Offline**\nCannot connect to `{API_BASE_URL}`.\nRun: `uvicorn main:app --reload`")

    st.markdown("---")
    page_nav = st.radio(
        "Navigation",
        ["💬 Ask Questions (Chat)", "📁 Manage & Ingest Documents"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("📚 Quick Document List")
    docs = fetch_documents()
    if docs:
        for d in docs:
            st.markdown(
                f"""
                <div class="doc-sidebar-badge">
                    <strong>{d['source']}</strong><br/>
                    <span style="color: #94A3B8; font-size: 0.76rem;"><code>{d['chunks']} chunks</code> • ID: <code>{d['doc_id']}</code></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("")
        if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
            if clear_all_documents():
                st.toast("Cleared all documents from vector store!", icon="🗑️")
                st.rerun()
            else:
                st.error("Failed to clear documents.")
    else:
        st.info("No documents currently indexed.")


# Main App Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">
            <span class="pulse-dot"></span>
            <span>Self-Corrective RAG Pipeline</span>
        </div>
        <h1 class="hero-title"> DocuFlow AI</h1>
        <p class="hero-subtitle">Interactive AI intelligence powered by LangGraph, ChromaDB, and FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if page_nav == "💬 Ask Questions (Chat)":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("<p style='font-weight: 600; color: #CBD5E1; margin-bottom: 0.5rem;'>💡 Suggested Questions:</p>", unsafe_allow_html=True)
    sample_cols = st.columns(4)
    sample_prompts = [
        "How do I add a conditional edge in LangGraph?",
        "What is a StateGraph and how do nodes work?",
        "How does ChromaDB persist embeddings to disk?",
        "Explain quantum entanglement in physics.",
    ]

    selected_prompt = None
    for i, prompt in enumerate(sample_prompts):
        with sample_cols[i]:
            if st.button(prompt, key=f"sample_{i}", use_container_width=True):
                selected_prompt = prompt

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]

                with st.expander("🔍 Pipeline Trace & Citations"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.caption(f"**Query Type:** `{meta.get('query_type')}`")
                    with col_b:
                        st.caption(f"**Retries Used:** `{meta.get('retry_count')}`")
                    with col_c:
                        status = "✅ Grounded" if meta.get("generation_supported") else "⚠️ Fallback"
                        st.caption(f"**Status:** {status}")

                    st.markdown(f"**Rewritten Query:** `{meta.get('rewritten_query')}`")

                    sources = meta.get("sources", [])
                    if sources:
                        st.markdown("**Citations:**")
                        for s in sources:
                            heading = f" — *{s['heading']}*" if s.get("heading") else ""
                            st.markdown(f"- 📄 `{s['source']}`{heading} (`{s['chunk_id']}`)")
                    else:
                        st.info("No source chunks utilized (fallback path triggered).")

                if f"feedback_{idx}" not in st.session_state:
                    fb_col1, _ = st.columns([1, 4])
                    with fb_col1:
                        rating = st.feedback("thumbs", key=f"fb_widget_{idx}")
                        if rating is not None:
                            rating_str = "thumbs_up" if rating == 1 else "thumbs_down"
                            if submit_feedback(rating_str, "", message.get("question", "")):
                                st.session_state[f"feedback_{idx}"] = rating_str
                                st.toast("Feedback recorded to data/feedback.jsonl!", icon="✅")
                                st.rerun()

    user_query = st.chat_input("Ask a question about the indexed documentation...") or selected_prompt

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, retrieving chunks & grading relevance..."):
                try:
                    response = submit_query(user_query)
                    answer = response.get("answer", "No answer returned.")
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "question": user_query,
                        "metadata": {
                            "query_type": response.get("query_type"),
                            "rewritten_query": response.get("rewritten_query"),
                            "retry_count": response.get("retry_count"),
                            "sources": response.get("sources", []),
                            "generation_supported": response.get("generation_supported"),
                        },
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to generate answer: {str(e)}")


elif page_nav == "📁 Manage & Ingest Documents":
    st.subheader("📁 Ingest New Documentation")
    st.markdown("<p style='color: #94A3B8; margin-top: -0.5rem; margin-bottom: 1.5rem;'>Upload documents or enter URLs to index them into ChromaDB.</p>", unsafe_allow_html=True)

    upload_col, url_col = st.columns(2)

    with upload_col:
        st.markdown("#### 📄 Upload Local Files")
        uploaded_files = st.file_uploader(
            "Select documentation files (`.md`, `.txt`, `.html`, `.pdf`)",
            type=["md", "txt", "html", "pdf"],
            accept_multiple_files=True,
            key="file_uploader_widget",
        )

        if uploaded_files:
            st.caption(f"Selected {len(uploaded_files)} file(s): " + ", ".join([f.name for f in uploaded_files]))

            if st.button("🚀 Ingest Selected Files", type="primary", use_container_width=True):
                with st.spinner("Processing documents, splitting into chunks & generating embeddings..."):
                    try:
                        result = upload_files(uploaded_files)
                        added = result.get("total_chunks_added", 0)

                        if added > 0:
                            st.success(f"🎉 Successfully ingested! Added **{added} chunks** into ChromaDB.")
                        else:
                            st.warning("No new chunks added. (Files might already be indexed or contain duplicate content).")

                        for item in result.get("ingested", []):
                            st.markdown(f"- **{item['source']}**: `{item['chunks_added']} added`, `{item['chunks_skipped_duplicate']} skipped (duplicates)`")

                        if result.get("errors"):
                            for err in result["errors"]:
                                st.error(f"❌ Error in {err['source']}: {err['error']}")

                    except Exception as ex:
                        st.error(f"Upload failed: {str(ex)}")

    with url_col:
        st.markdown("#### 🌐 Ingest from URL")
        input_url = st.text_input(
            "Enter raw document or web documentation URL",
            placeholder="https://raw.githubusercontent.com/.../README.md",
        )

        if st.button("📥 Fetch & Ingest URL", use_container_width=True):
            if not input_url:
                st.warning("Please enter a URL first.")
            else:
                with st.spinner("Fetching URL and generating embeddings..."):
                    try:
                        result = ingest_url(input_url)
                        added = result.get("total_chunks_added", 0)
                        if added > 0:
                            st.success(f"🎉 Added **{added} chunks** from URL!")
                        else:
                            st.warning("No new chunks added (duplicate content).")

                        for err in result.get("errors", []):
                            st.error(f"❌ {err['source']}: {err['error']}")

                    except Exception as ex:
                        st.error(f"URL Ingestion failed: {str(ex)}")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📋 Currently Indexed Documents")
    current_docs = fetch_documents()

    if current_docs:
        for doc in current_docs:
            d_col1, d_col2, d_col3 = st.columns([3, 1, 1])
            with d_col1:
                st.markdown(f"**{doc['source']}** (`{doc['doc_id']}`)")
                if doc.get("title"):
                    st.caption(f"Title: {doc['title']}")
            with d_col2:
                st.markdown(f"`{doc['chunks']} chunks`")
            with d_col3:
                if st.button("Delete", key=f"del_{doc['doc_id']}"):
                    if delete_single_document(doc["doc_id"]):
                        st.toast(f"Deleted {doc['source']}", icon="🗑️")
                        st.rerun()
                    else:
                        st.error("Delete failed.")
            st.divider()

        if st.button("🗑️ Wipe All Documents", type="secondary"):
            if clear_all_documents():
                st.toast("Cleared all documents!", icon="🗑️")
                st.rerun()
            else:
                st.error("Failed to clear documents.")
    else:
        st.info("The vector store is currently empty. Use the upload box above to ingest your first document!")
