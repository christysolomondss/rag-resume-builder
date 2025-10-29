# AI Agent Instructions for RAG Resume Builder

## Project Overview
This is a Streamlit-based RAG (Retrieval Augmented Generation) application that processes bulk resume PDFs and enables conversational queries about candidates. The app uses LangChain, OpenAI, and FAISS for document processing and retrieval.

## Key Components & Architecture
- `app.py`: Main application file containing all core functionality
- `resume_faiss_index/`: Directory for persisting FAISS vector store
- Core data flow:
  1. PDF upload → text extraction → candidate name inference
  2. Document embedding → FAISS indexing
  3. Query processing → RAG retrieval → LLM response generation

## Environment Setup
Required environment variables:
- `OPENAI_API_KEY`: Must be set in environment or `.env` file

Dependencies installation:
```sh
pip install -r requirements.txt
```

## Development Workflow
1. Launch application:
```sh
streamlit run app.py
```

2. Key configurations in `app.py`:
- `EMBEDDINGS_MODEL = "text-embedding-ada-002"`
- `LLM_MODEL = "gpt-4"` (or "gpt-3.5-turbo")
- `INDEX_DIR = "resume_faiss_index"`

## Project Conventions
1. Document Processing:
- Each PDF is treated as one document with metadata
- Candidate name inference uses a heuristic approach (see `infer_candidate_name()`)
- Document metadata structure: `{"source": filename, "candidate_name": inferred_name}`

2. State Management:
- Key Streamlit session state objects:
  - `st.session_state.doc_store`: FAISS vector store
  - `st.session_state.qa_chain`: RetrievalQA chain
  - `st.session_state.docs_metadata`: List of document metadata

3. Query Enhancement:
- Candidate filtering is implemented by prepending directives to queries
- Format: `"Candidate: {filter} Question: {query}"`

## Common Tasks
- Rebuild index: Use sidebar "Rebuild Index" button
- Clear index: Use sidebar "Clear Index" button
- Filter queries: Use "Candidate filter" field for candidate-specific questions

## Extension Points
1. Name Detection: Replace naive heuristic in `infer_candidate_name()` with NER
2. PDF Processing: Add support for proper resume separation in bulk PDFs
3. Query Chain: Customize `PROMPT` variable for specialized retrieval behavior