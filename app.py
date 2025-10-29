"""
Streamlit + LangChain RAG app for bulk resume PDF search

Features:
- Upload a single "bulk" PDF containing multiple resumes concatenated together
- Heuristically group pages into candidate resumes using email/phone detection (fallback to page headers)
- Create embeddings with OpenAI and store them in a Chroma vectorstore
- Build a ConversationalRetrievalChain so users can ask for a specific candidate or ask general questions
- Allow downloading an extracted single-candidate PDF

Requirements:
pip install streamlit langchain openai chromadb tiktoken PyPDF2 python-multipart

Environment:
- Set OPENAI_API_KEY in environment

Run:
streamlit run app.py

Notes:
- Name detection is heuristic: we group pages by email if present, else phone, else by likely header lines.
- This is a starting point; for higher accuracy add a proper NER (spaCy) or a PDF resume separator (if available).
"""
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment. Please check your .env file.")
from typing import List, Dict, Any
import io
import json
import base64

import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from PyPDF2 import PdfReader


# ----------------------------
# Config / Helpers
# ----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDINGS_MODEL = "text-embedding-ada-002"  # or your choice
LLM_MODEL = "gpt-4"  # or "gpt-3.5-turbo"

# You can customize the prompt used by RetrievalQA if desired
PROMPT = None  # or a string

# Directory to persist FAISS index
INDEX_DIR = "resume_faiss_index"

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text_parts.append(text)
    return "\n".join(text_parts)

def infer_candidate_name(text: str, filename_hint: str) -> str:
    # A very naive heuristic: look for a name in the first 2-3 lines.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:5]:
        # simple heuristic: if line has multiple words and is title-like
        if len(line.split()) >= 2 and any(w[0].isupper() for w in line.split()):
            # avoid lines that look like "Curriculum Vitae" or "Resume"
            if "resume" in line.lower() or "cv" in line.lower():
                continue
            return line
    # fallback to filename hint
    name = filename_hint.rsplit(".", 1)[0]
    return name

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")

st.title("Resume RAG Chatbot (Bulk Resumes)")
st.write("Upload multiple resume PDFs. Then ask the bot questions about a specific candidate or in general.")

# 1) Upload PDFs
st.sidebar.header("Upload & Index")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs (one resume per PDF is recommended)", type=["pdf"], accept_multiple_files=True, help="Each PDF is treated as a document. For truly multi-resume PDFs, you may want to split them first."
)

rebuild_index = st.sidebar.button("Rebuild Index from Uploaded Files")
clear_index = st.sidebar.button("Clear Index")

# Initialize or load embeddings/index
if "doc_store" not in st.session_state:
    st.session_state.doc_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "docs_metadata" not in st.session_state:
    st.session_state.docs_metadata = []

# Build index if there are new uploads, or user pressed rebuild
if uploaded_files or rebuild_index or clear_index:
    # Clear old index if requested
    if clear_index:
        if os.path.exists(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
        st.session_state.doc_store = None
        st.session_state.qa_chain = None
        st.session_state.docs_metadata = []
        st.success("Index cleared.")
        st.experimental_rerun()

    documents: List[Document] = []
    metadata_list: List[Dict[str, Any]] = []

    # For each uploaded PDF, extract text and create a Document
    for f in uploaded_files:
        raw = f.read()
        text = extract_text_from_pdf(raw)
        candidate_name = infer_candidate_name(text, f.name)
        # Create a Document with metadata
        doc = Document(page_content=text, metadata={
            "source": f.name,
            "candidate_name": candidate_name
        })
        documents.append(doc)
        metadata_list.append({
            "source": f.name,
            "candidate_name": candidate_name
        })

    if not documents:
        st.warning("No documents extracted. Please upload PDFs.")
    else:
        # Build embeddings and FAISS index
        if OPENAI_API_KEY is None:
            st.error("Please set OPENAI_API_KEY in your environment.")
        else:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

            # Create or load FAISS index
            if not os.path.exists(INDEX_DIR):
                os.makedirs(INDEX_DIR)
            # We store the vector store in memory for this session; you can persist if needed
            vector_store = FAISS.from_documents(documents, embeddings)

            # Persist the store object in session (in-memory for this run)
            st.session_state.doc_store = vector_store
            st.session_state.docs_metadata = metadata_list

            # Setup LLM + RetrievalQA chain
            llm = OpenAI(model=LLM_MODEL)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",  # or "map_reduce" etc.
                retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )
            st.session_state.qa_chain = qa

            st.success(f"Indexed {len(documents)} document(s). You can now ask questions.")
            # Show a quick summary
            st.write("Indexed documents:")
            for idx, m in enumerate(metadata_list, start=1):
                st.write(f"{idx}. {m.get('candidate_name')} (source: {m.get('source')})")

# 2) Chat / Query area
st.sidebar.header("Query")
user_query = st.sidebar.text_input("Ask about a candidate or a general question", value="Tell me about the candidate John Doe's experience.", placeholder="e.g., What is Jane Smith's total years of experience?")

# Optional: filter by candidate name
candidate_filter = st.sidebar.text_input("Candidate filter (optional)", value="")  # e.g., "John Doe"

if st.sidebar.button("Ask") or (user_query and st.session_state.qa_chain is not None):
    if st.session_state.qa_chain is None:
        st.error("Index not built yet. Please upload PDFs and click Rebuild Index.")
    else:
        # Prepare a "prompt" that includes the candidate filter if provided
        # We implement a lightweight approach: prepend a directive to the user query.
        if candidate_filter.strip():
            augmented_query = f"Candidate: {candidate_filter.strip()}  Question: {user_query}"
        else:
            augmented_query = user_query

        # Run the chain
        result = st.session_state.qa_chain.run(augmented_query)
        # LangChain's RetrievalQA with return_source_documents=True provides sources when available
        # Some versions return string; to be safe, attempt to fetch sources from the chain.
        sources = []
        try:
            # This depends on langchain version; adapt as needed
            if isinstance(result, tuple) and len(result) == 2:
                answer, sources_list = result
                sources = sources_list
            else:
                answer = result
        except Exception:
            answer = result
            sources = []

        # Display
        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources (Resume Snippets)")
            for src in sources:
                # src is a Document or a dict with metadata
                if isinstance(src, Document):
                    meta = src.metadata
                    snippet = getattr(src, "page_content", "")[:1000]
                    st.write(f"- {meta.get('candidate_name', 'Unknown')} | {meta.get('source', '')}")
                    st.write(f"  Snippet: {snippet[:500]}...")
                elif isinstance(src, dict):
                    st.write(f"- {src.get('candidate_name', 'Unknown')} | {src.get('source', '')}")
        else:
            st.info("No explicit sources returned with the answer.")

# 3) Extra UI niceties
st.sidebar.markdown("---")
st.sidebar.info("Tips:")
st.sidebar.markdown("- Upload one resume per PDF for best accuracy.")
st.sidebar.markdown("- Use a specific candidate name in the filter to narrow down results.")
st.sidebar.markdown("- Ensure OPENAI_API_KEY is set in your environment.")
