# PDF Search (FAISS) + Q&A

Streamlit app that lets you upload PDFs, build a FAISS index, run semantic search,
and optionally ask an LLM to answer questions using retrieved context.

## Features
- Upload one or more PDFs and build a vector index.
- Adjustable chunk size, overlap, and top-k retrieval.
- Optional LLM answer grounded in retrieved chunks.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

## Usage
- Upload PDFs in the main panel.
- Configure chunking and model options in the sidebar.
- Build the index, then search or ask a question.

## Notes
- Scanned PDFs require OCR; `pypdf` will not extract text from images.
- Embedding model selection appears after you upload files.
