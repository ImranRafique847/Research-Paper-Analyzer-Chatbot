# app.py
import streamlit as st
from dotenv import load_dotenv
from rag_chatbot import process_pdf, retrieve_chunks, query_llama
import time

load_dotenv()

# ----------------- Streamlit UI config -----------------
st.set_page_config(page_title="Research Paper RAG Chatbot", layout="wide")
st.title("📘 Research Paper Analyzer RAG Chatbot")
st.write(
    "Flow: ask → get answer → new input appears below (you can optionally upload/replace a PDF for the next question). "
    "Switch on 'Use PDF context' to include paper context; if no relevant context exists the model will still answer."
)

# ----------------- Session state initialization -----------------
if "qa_history" not in st.session_state:
    # each element is dict: {'question': str, 'answer': str, 'pdf': str or None}
    st.session_state.qa_history = []

if "vector_stores" not in st.session_state:
    # map: filename -> FAISS index
    st.session_state.vector_stores = {}

if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

# Show conversation history (top → older first)
for i, pair in enumerate(st.session_state.qa_history):
    q = pair["question"]
    a = pair["answer"]
    pdf = pair.get("pdf")
    st.markdown(f"**🧑 You**  {(f'— ({pdf})' if pdf else '')}: {q}")
    st.markdown(f"**🤖 Assistant:** {a}")
    st.markdown("---")

# ----------------- New-question form (always a single form at bottom) -----------------
st.markdown("### 💬 Ask a new question")
with st.form(key="qa_form", clear_on_submit=True):
    user_question = st.text_area("Your question", height=120, placeholder="e.g. Summarize the abstract or explain the method...")
    # If there are already uploaded PDFs, offer a selector (optional)
    if st.session_state.vector_stores:
        pdf_list = list(st.session_state.vector_stores.keys())
        # show a selectbox defaulting to last used or first item
        default_index = 0
        if st.session_state.selected_pdf in pdf_list:
            default_index = pdf_list.index(st.session_state.selected_pdf)
        chosen_pdf = st.selectbox("Choose existing PDF to use (optional)", pdf_list, index=default_index)
    else:
        chosen_pdf = None

    # Upload a new PDF (ONLY inside this form; avoids duplicate upload widgets)
    new_pdf = st.file_uploader("Or upload a new PDF (optional)", type=["pdf"], key="uploader")

    # Toggle whether to use PDF context (if false, no retrieval used)
    use_pdf_context = st.checkbox("Use PDF context (RAG)", value=True)

    submitted = st.form_submit_button("Send")

    if submitted and user_question:
        # 1) If a new PDF provided -> process it (progress shown inside process_pdf)
        pdf_to_use = None
        if new_pdf is not None:
            # Save and process (this shows a progress bar and status)
            with st.spinner("Processing uploaded PDF..."):
                vs = process_pdf(new_pdf, new_pdf.name)
            if vs:
                st.session_state.vector_stores[new_pdf.name] = vs
                st.session_state.selected_pdf = new_pdf.name
                pdf_to_use = new_pdf.name
            else:
                # processing failed; still proceed without PDF
                pdf_to_use = None
        else:
            # no new upload: use chosen existing one (if any)
            pdf_to_use = chosen_pdf if chosen_pdf else st.session_state.selected_pdf

        # 2) Retrieve chunks if user asked to use PDF context and vectorstore exists
        retrieved_chunks = []
        if use_pdf_context and pdf_to_use and pdf_to_use in st.session_state.vector_stores:
            vectorstore = st.session_state.vector_stores[pdf_to_use]
            # use top-k retrieval; FAISS will return up to k items (if available)
            retrieved_chunks = retrieve_chunks(vectorstore, user_question, k=5)

        # 3) Build final prompt (we recommend including context and instruction to fallback to knowledge if needed)
        # You may prefer to send context + question separately to your LLaMA. This concatenation is a simple default:
        prompt = ""
        if retrieved_chunks:
            prompt += (
                "You are a research assistant. Use the context below (from the uploaded paper) to answer the question. "
                "If the context does not contain the answer, use your general knowledge to answer the question.\n\n"
                "CONTEXT:\n"
                + "\n\n".join(retrieved_chunks)
                + "\n\nQUESTION:\n"
                + user_question
            )
        else:
            # no context found or not using PDF — just ask normally, but keep instruction style consistent
            prompt = f"You are a helpful research assistant. Answer the question:\n\n{user_question}"

        # 4) Call the LLaMA inference (replace query_llama with your actual inference call)
        with st.spinner("Generating answer..."):
            answer = query_llama(prompt, retrieved_chunks=retrieved_chunks, history=[(p["question"], p["answer"]) for p in st.session_state.qa_history])

        # 5) Save history and update selected_pdf
        st.session_state.qa_history.append({"question": user_question, "answer": answer, "pdf": pdf_to_use})
        if pdf_to_use:
            st.session_state.selected_pdf = pdf_to_use

        # Rerun to show the updated history and new input box below
        st.rerun()
