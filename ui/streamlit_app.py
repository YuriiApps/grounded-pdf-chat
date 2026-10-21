import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Grounded PDF Chat", page_icon="📄")
st.title("Grounded PDF Chat")
st.caption("Answers come only from the indexed PDFs, with document and page citations.")

with st.sidebar:
    st.subheader("Index")
    if st.button("Re-index PDFs"):
        with st.spinner("Indexing…"):
            r = requests.post(f"{API_URL}/ingest", timeout=600)
        if r.ok:
            data = r.json()
            st.success(f"{len(data['files'])} files, {data['total_chunks']} chunks")
        else:
            st.error(r.text)

if "history" not in st.session_state:
    st.session_state.history = []


def show(entry: dict) -> None:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            with st.expander("Sources"):
                for src in entry["sources"]:
                    st.markdown(f"**[{src['ref']}] {src['source']}, page {src['page']}** · score {src['score']}")
                    st.caption(src["snippet"])


for entry in st.session_state.history:
    show(entry)

if question := st.chat_input("Ask about the documents"):
    try:
        r = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=120)
        r.raise_for_status()
        entry = r.json()
    except requests.RequestException as exc:
        entry = {"question": question, "answer": f"API error: {exc}", "found": False, "sources": []}
    st.session_state.history.append(entry)
    show(entry)
