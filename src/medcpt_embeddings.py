"""MedCPT embeddings: NCBI's contrastively-trained biomedical retriever.

Unlike a general sentence-transformer (one shared encoder for both sides),
MedCPT is asymmetric — a separate Query-Encoder and Article-Encoder, trained
on 255M real PubMed search-log query/article pairs. Both project into the
same 768-dim space, so similarity search still works normally; we just have
to route queries and documents through the right model.
"""

from typing import List

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"


def _cls_embed(model, tokenizer, inputs, max_length: int, batch_size: int) -> List[List[float]]:
    """Embed inputs in small batches, taking the CLS token as the pooled vector."""
    all_embeddings: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            output = model(**encoded)
            cls_embeddings = output.last_hidden_state[:, 0, :]
            all_embeddings.extend(cls_embeddings.tolist())
    return all_embeddings


class MedCPTEmbeddings(Embeddings):
    """LangChain-compatible Embeddings backed by MedCPT's dual encoders."""

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        self.query_model = AutoModel.from_pretrained(QUERY_MODEL_NAME).eval()
        self.article_tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL_NAME)
        self.article_model = AutoModel.from_pretrained(ARTICLE_MODEL_NAME).eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # The article encoder expects [title, text] pairs; we have no titles
        # for arbitrary PDF chunks, so pass an empty title.
        pairs = [["", text] for text in texts]
        return _cls_embed(
            self.article_model,
            self.article_tokenizer,
            pairs,
            max_length=512,
            batch_size=self.batch_size,
        )

    def embed_query(self, text: str) -> List[float]:
        return _cls_embed(
            self.query_model,
            self.query_tokenizer,
            [text],
            max_length=64,
            batch_size=1,
        )[0]
