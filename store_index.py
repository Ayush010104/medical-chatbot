"""Build (or refresh) the Pinecone index from PDFs placed in data/."""

import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings, load_pdf_files, text_split

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
EMBEDDING_DIMENSION = 384  # output size of all-MiniLM-L6-v2


def main() -> None:
    print("Loading PDFs from data/ ...")
    documents = load_pdf_files("data/")
    print(f"Loaded {len(documents)} document(s).")

    chunks = text_split(documents)
    print(f"Split into {len(chunks)} chunk(s).")

    embeddings = download_hugging_face_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    print(f"Upserting {len(chunks)} chunk(s) into '{INDEX_NAME}' ...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print("Done.")


if __name__ == "__main__":
    main()
