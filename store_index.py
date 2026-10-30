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

from src.helper import (
    download_hugging_face_embeddings,
    load_pdf_files,
    load_text_files,
    text_split,
)
from src.multimodal import build_image_documents, extract_tables

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
EMBEDDING_DIMENSION = 768  # output size of MedCPT's encoders
# Each department gets its own source folder and its own Pinecone namespace
# within the same index, so departments stay isolated without needing a
# separate (and, on the free tier, unavailable) index per department.
DATA_DIR = os.environ.get("DATA_DIR", "data/")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "")
IMAGE_OUTPUT_DIR = os.environ.get("IMAGE_OUTPUT_DIR", "extracted_images/")
# Image captioning makes one paid OpenAI vision call per embedded image, which
# is slow and costs money on a large PDF — opt in once the rest is verified.
ENABLE_IMAGE_CAPTIONING = os.environ.get("ENABLE_IMAGE_CAPTIONING", "false").lower() == "true"


def build_all_documents():
    print(f"Loading PDF text from {DATA_DIR} ...")
    text_documents = load_pdf_files(DATA_DIR)
    print(f"  -> {len(text_documents)} PDF page(s).")

    print(f"Loading plain-text sources from {DATA_DIR} ...")
    text_documents += load_text_files(DATA_DIR)

    text_chunks = text_split(text_documents)
    print(f"  -> {len(text_chunks)} text chunk(s) total.")

    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    print("Extracting tables ...")
    table_documents = []
    for pdf_path in pdf_paths:
        table_documents.extend(extract_tables(pdf_path))
    print(f"  -> {len(table_documents)} table(s).")

    image_documents = []
    if ENABLE_IMAGE_CAPTIONING:
        print("Extracting and captioning images ...")
        vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        for pdf_path in pdf_paths:
            image_documents.extend(
                build_image_documents(pdf_path, IMAGE_OUTPUT_DIR, vision_llm)
            )
        print(f"  -> {len(image_documents)} image(s).")
    else:
        print("Skipping image captioning (ENABLE_IMAGE_CAPTIONING=false).")

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

    namespace_label = NAMESPACE or "(default)"
    print(f"Upserting {len(all_documents)} document(s) into '{INDEX_NAME}' / namespace '{namespace_label}' ...")
    PineconeVectorStore.from_documents(
        documents=all_documents,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE or None,
    )
    print("Done.")


if __name__ == "__main__":
    main()
