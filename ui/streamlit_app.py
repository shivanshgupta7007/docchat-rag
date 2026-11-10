"""Streamlit UI: upload a PDF on the left, chat with it on the right.
Run with: streamlit run ui/streamlit_app.py
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag import RAGPipeline  # noqa: E402

st.set_page_config(page_title="DocChat", page_icon="📄")
st.title("DocChat - chat with your PDFs")

@st.cache_resource
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()

pipeline = load_pipeline()

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Index document"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        with st.spinner("Chunking, embedding, indexing..."):
            added = pipeline.ingest_pdf(tmp_path, source_name=uploaded.name)
        os.unlink(tmp_path)
        st.success(f"Indexed {added} chunks from {uploaded.name}")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask about your documents")
if question:
    with st.spinner("Thinking..."):
        result = pipeline.ask(question)
    st.session_state.history.append((question, result))

for q, r in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(r["answer"])
        with st.expander("Sources"):
            for s in r["sources"]:
                page = f", page {s['page']}" if s["page"] else ""
                st.caption(f"{s['source']}{page} (similarity {s['similarity']})")
                st.text(s["snippet"] + "...")
