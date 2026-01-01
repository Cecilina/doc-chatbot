import io
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import streamlit as st

load_dotenv()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts).strip()


def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_docs_from_pdfs(pdfs: List[Tuple[str, bytes]]) -> List[Document]:
    docs: List[Document] = []
    for filename, b in pdfs:
        text = extract_text_from_pdf(b)
        if not text:
            continue
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            docs.append(
                Document(page_content=ch, metadata={"source": filename, "chunk": i})
            )
    return docs


def run_app() -> None:
    st.title("PDF Search (FAISS) + Q&A")
    st.caption(
        "Upload PDFS -> build FAISS index"
        "-> semantic search -> optional LLM answer."
    )

    # -----------------------
    # Sidebar controls
    # -----------------------
    with st.sidebar:
        st.header("Index settings")
        chunk_size = st.slider("Chunk size", 300, 2500, 1000, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, 10)
        k = st.slider("Top-K results", 1, 12, 5)
        st.divider()

        st.header("LLM settings (optional)")
        use_llm = st.checkbox("Generate an answer with an LLM", value=False)
        model_name = st.text_input(
            "Model",
            value="gpt-4o-mini",
            disabled=not use_llm,
        )

    # -----------------------
    # Session state
    # -----------------------
    if "db" not in st.session_state:
        st.session_state.db = None
    if "docs_count" not in st.session_state:
        st.session_state.docs_count = 0
    if "files" not in st.session_state:
        st.session_state.files = []

    # ----------------------
    # Upload
    # ----------------------
    embedding_model = "text-embedding-3-small"
    uploaded = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        with st.sidebar:
            embedding_model = st.selectbox(
                "Embedding model",
                options=[
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                ],
                index=0,
            )

    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        if uploaded:
            st.write("**Selected files:**")
            for f in uploaded:
                st.write(f"- {f.name} ({f.size:,} bytes)")
            if st.button("Build / Rebuild index", type="primary"):
                pdfs = [(f.name, f.read()) for f in uploaded]

                # Use sidebar chunking values

                docs = build_docs_from_pdfs(pdfs)

                if not docs:
                    st.error(
                        "No extractable text found."
                        "If these are scanned PDFs,"
                        "you'll need OCR."
                    )
                else:
                    embeddings = OpenAIEmbeddings(model=embedding_model)
                    db = FAISS.from_documents(docs, embeddings)

                    st.session_state.db = db
                    st.session_state.docs_count = len(docs)
                    st.session_state.files = [f.name for f in uploaded]
                    st.success(
                        f"Built FAISS index with {len(docs)}"
                        f"chunks from {len(uploaded)} PDF(s)."
                    )

            else:
                st.info("Upload PDFs to get started.")

    with colB:
        st.subheader("Index status")
        if st.session_state.db is None:
            st.write("No index yet.")
        else:
            st.write(f"- Files: {', '.join(st.session_state.files)}")
            st.write(f"- Chunks indexed: {st.session_state.docs_count}")

    # ---------------------------
    # Query
    # ---------------------------
    st.divider()
    query = st.text_input(
        "Ask a question or search your PDFs",
        placeholder="e.g., What are the key takeaways?",
    )

    if st.button("Search", disabled=st.session_state.db is None or not query):
        db: FAISS = st.session_state.db
        results = db.similarity_search(query, k=k)

        st.subheader("Top matches")
        for i, doc in enumerate(results, start=1):
            src = doc.metadata.get("source", "unknown")
            ch = doc.metadata.get("chunk", "?")
            with st.expander(f"{i}. {src} - chunk {ch}", expanded=(i == 1)):
                st.write(doc.page_content)

        if use_llm:
            st.subheader("LLM answer")
            llm = ChatOpenAI(model=model_name, temperature=0)

            context = "\n\n----\n\n".join(
                f"Source: {d.metadata.get('source')}"
                f"(chunk {d.metadata.get('chunk')}) \n"
                f"{d.page_content}"
                for d in results
            )

            prompt = (
                "Answer the user's question using ONLY the context below. "
                "If the answer isn't in the context, say you don't know.\n\n"
                f"Question: {query}\n\n"
                f"Context:\n{context}"
            )

            resp = llm.invoke(prompt)
            st.write(resp.content)

    st.caption(
        "Tip: if your PDFs are scanned images,"
        "use OCR first (PyPDF2/pypdf won't extract text)"
    )


if __name__ == "__main__":
    run_app()
