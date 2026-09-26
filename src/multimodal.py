"""Table and image extraction from PDFs, so the RAG index isn't text-only.

Tables are converted to Markdown text (same embedding model, no new vector
space needed). Images are extracted to disk and captioned by a vision LLM;
the caption text is what gets embedded and searched, while the image file
itself is kept so the UI can show/cite it.
"""

import base64
import os
from typing import List

import pymupdf as fitz
import pdfplumber
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


def _rows_to_markdown(rows: List[List[str]]) -> str:
    cleaned = [[cell.strip() if cell else "" for cell in row] for row in rows]
    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_tables(pdf_path: str) -> List[Document]:
    """Extract each detected table as one Document, rendered as Markdown."""
    documents: List[Document] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue
                markdown = _rows_to_markdown(table)
                documents.append(
                    Document(
                        page_content=markdown,
                        metadata={
                            "source": pdf_path,
                            "page": page_number,
                            "content_type": "table",
                        },
                    )
                )
    return documents


def extract_images(pdf_path: str, output_dir: str) -> List[dict]:
    """Extract embedded images to output_dir. Returns file path + page metadata."""
    os.makedirs(output_dir, exist_ok=True)
    extracted = []
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    doc = fitz.open(pdf_path)
    try:
        for page_number in range(len(doc)):
            page = doc[page_number]
            for image_index, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                extension = base_image["ext"]

                filename = f"{pdf_stem}_p{page_number + 1}_{image_index}.{extension}"
                output_path = os.path.join(output_dir, filename)
                with open(output_path, "wb") as f:
                    f.write(image_bytes)

                extracted.append(
                    {
                        "path": output_path,
                        "source": pdf_path,
                        "page": page_number + 1,
                    }
                )
    finally:
        doc.close()

    return extracted


CAPTION_PROMPT = (
    "Describe this image from a medical reference document factually and "
    "concisely, in 1-3 sentences. Note any visible labels, anatomical "
    "structures, or findings. This description will be used to help retrieve "
    "the image for relevant search queries, not to diagnose anything."
)


def caption_image(image_path: str, vision_llm) -> str:
    """Generate a text caption for one image using a vision-capable chat model."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    extension = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if extension in ("jpg", "jpeg") else extension

    message = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{image_b64}"},
            },
        ]
    )
    response = vision_llm.invoke([message])
    return response.content.strip()


def build_image_documents(pdf_path: str, output_dir: str, vision_llm) -> List[Document]:
    """Extract every image in pdf_path and turn it into a captioned Document."""
    documents = []
    for image in extract_images(pdf_path, output_dir):
        caption = caption_image(image["path"], vision_llm)
        documents.append(
            Document(
                page_content=caption,
                metadata={
                    "source": image["source"],
                    "page": image["page"],
                    "content_type": "image",
                    "image_path": image["path"],
                },
            )
        )
    return documents
