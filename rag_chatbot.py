# rag_chatbot.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from typing import List

# ---------- CPU-safe sentence-transformers wrapper ----------
class CustomHFEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: List[str]):
        arr = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return arr.tolist()

    def embed_query(self, text: str):
        arr = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return arr[0].tolist()

    # 👇 Add this to make it compatible with FAISS
    def __call__(self, text: str):
        return self.embed_query(text)

# ---------- Process PDF (writes file, shows progress, returns FAISS index) ----------
def process_pdf(pdf_file, file_name: str):
    """
    Save uploaded file to ./data/, parse pages with PyPDFLoader, split to chunks,
    build FAISS using CPU-safe embeddings. Shows progress bar & status in Streamlit.
    Returns: FAISS vectorstore or None on failure.
    """
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", file_name)
    # Save file bytes
    with open(save_path, "wb") as f:
        f.write(pdf_file.read())

    # Load pages
    loader = PyPDFLoader(save_path)
    docs = loader.load()
    total_pages = len(docs)
    if total_pages == 0:
        st.error("❌ PDF has no pages or could not be read.")
        return None

    progress_bar = st.progress(0)
    status = st.empty()

    # Split pages into chunks (keeps metadata like page number)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = []
    for i, doc in enumerate(docs):
        chunks = splitter.split_documents([doc])
        split_docs.extend(chunks)
        pct = int(((i + 1) / total_pages) * 100)
        progress_bar.progress(pct)
        status.text(f"📄 Processing page {i+1}/{total_pages} ({pct}%)")

    status.text("⚡ Building embeddings and index (this may take a few seconds)...")

    # Create embeddings (CPU-safe)
    embeddings = CustomHFEmbeddings()
    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
    except Exception as e:
        st.error(f"❌ Error while building vectorstore: {e}")
        progress_bar.empty()
        status.empty()
        return None

    progress_bar.empty()
    status.text("✅ PDF processed and indexed successfully!")
    return vectorstore

# ---------- Retriever helper ----------
def retrieve_chunks(vectorstore, query: str, k: int = 5):
    """Return list of page_content strings (top k)."""
    if not vectorstore:
        return []
    docs = vectorstore.similarity_search(query, k=k)
    return [d.page_content for d in docs]

import requests
import os

def query_llama(prompt: str, retrieved_chunks: list = None, history: list = None):
    """
    Query LLaMA model hosted on Groq API.
    Make sure GROQ_API_KEY is in your .env file.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return "❌ Missing GROQ_API_KEY in .env file."

    # Build context
    ctx = "\n\n".join(retrieved_chunks) if retrieved_chunks else ""
    messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {prompt}"}
    ]

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",  # or "llama-3.1-70b-versatile"
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 500,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ LLaMA API request failed: {e}"
