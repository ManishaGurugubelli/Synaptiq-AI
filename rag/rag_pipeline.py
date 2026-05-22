from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from functools import lru_cache


# -----------------------------
# Load Embedding Model
# -----------------------------

@lru_cache(maxsize=1)
def load_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# -----------------------------
# Load PDF
# -----------------------------

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


# -----------------------------
# Split Documents
# -----------------------------

def split_documents(documents):

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# -----------------------------
# Create Vector Store
# -----------------------------

def create_vectorstore(chunks):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectorstore = FAISS.from_texts(
        texts,
        load_embedding_model()
    )

    return vectorstore


# -----------------------------
# Save Vector Store
# -----------------------------

def save_vectorstore(vectorstore):

    vectorstore.save_local(
        "vectorstore/faiss_index"
    )


# -----------------------------
# Process PDF
# -----------------------------

def process_pdf(pdf_path):

    documents = load_pdf(
        pdf_path
    )

    chunks = split_documents(
        documents
    )

    vectorstore = create_vectorstore(
        chunks
    )

    save_vectorstore(
        vectorstore
    )

    return (
        f"PDF processed successfully! "
        f"{len(chunks)} chunks created."
    )


# -----------------------------
# Load Vector Store (Cached)
# -----------------------------

@lru_cache(maxsize=1)
def load_vectorstore():

    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        load_embedding_model(),
        allow_dangerous_deserialization=True
    )

    return vectorstore


# -----------------------------
# Retrieve Context
# -----------------------------

def retrieve_context(query):

    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(
        query,
        k=2
    )

    context_list = []

    for doc in docs:

        page = doc.metadata.get(
            "page",
            None
        )

        text = doc.page_content

        # Safe page handling
        if isinstance(page, int):

            page_info = f"Page {page + 1}"

        else:

            page_info = "Page Unknown"

        context_list.append(
            f"[{page_info}]\n{text}"
        )

    context = "\n\n".join(
        context_list
    )

    return context