"""Build (or refresh) the Pinecone index from PDFs placed in data/.

Indexes three content types per PDF, all embedded through the same model
into the same index so a single similarity search covers all of them:
  - text        (existing prose, chunked as before)
  - table       (detected tables, rendered as Markdown)
  - image       (embedded images, captioned by a vision LLM)
"""

import glob
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings, load_pdf_files, text_split
from src.multimodal import build_image_documents, extract_tables

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
EMBEDDING_DIMENSION = 384  # output size of all-MiniLM-L6-v2
DATA_DIR = "data/"
IMAGE_OUTPUT_DIR = os.environ.get("IMAGE_OUTPUT_DIR", "extracted_images/")


def build_all_documents():
    print("Loading PDF text from data/ ...")
    text_documents = load_pdf_files(DATA_DIR)
    text_chunks = text_split(text_documents)
    print(f"  -> {len(text_chunks)} text chunk(s).")

    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    print("Extracting tables ...")
    table_documents = []
    for pdf_path in pdf_paths:
        table_documents.extend(extract_tables(pdf_path))
    print(f"  -> {len(table_documents)} table(s).")

    print("Extracting and captioning images ...")
    vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    image_documents = []
    for pdf_path in pdf_paths:
        image_documents.extend(
            build_image_documents(pdf_path, IMAGE_OUTPUT_DIR, vision_llm)
        )
    print(f"  -> {len(image_documents)} image(s).")

    return text_chunks + table_documents + image_documents


def main() -> None:
    all_documents = build_all_documents()
    print(f"Total documents to index: {len(all_documents)}")

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

    print(f"Upserting {len(all_documents)} document(s) into '{INDEX_NAME}' ...")
    PineconeVectorStore.from_documents(
        documents=all_documents,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print("Done.")


if __name__ == "__main__":
    main()
