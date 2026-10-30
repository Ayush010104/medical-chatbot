from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.medcpt_embeddings import MedCPTEmbeddings


def load_pdf_files(data_dir: str):
    """Load every PDF found in data_dir into LangChain Document objects."""
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def load_text_files(data_dir: str):
    """Load every .txt found in data_dir into LangChain Document objects."""
    loader = DirectoryLoader(
        data_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
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
    """Return the embedding model used for indexing and querying (MedCPT, biomedical-domain)."""
    return MedCPTEmbeddings()
