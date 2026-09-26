from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_files(data_dir: str):
    """Load every PDF found in data_dir into LangChain Document objects."""
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def text_split(documents, chunk_size: int = 500, chunk_overlap: int = 20):
    """Split documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata.setdefault("content_type", "text")
    return chunks


def download_hugging_face_embeddings():
    """Return the sentence-transformers embedding model used for indexing and querying."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
