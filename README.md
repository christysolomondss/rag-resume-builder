# rag-resume-builder# Resume RAG Chatbot

A Streamlit application that uses LangChain and OpenAI to process and analyze resumes.

## Setup

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AI_DEMO.git
cd AI_DEMO
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a .env file with your OpenAI API key
```bash
OPENAI_API_KEY=your_api_key_here
```

5. Run the application
```bash
streamlit run app.py
```

## Features

- Upload multiple resume PDFs
- Extract and index resume content
- Query resumes using natural language
- Filter results by candidate

## Technologies

- Streamlit
- LangChain
- OpenAI
- FAISS
- ChromaDB
- PyPDF2
