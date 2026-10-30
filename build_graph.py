"""Extract oncology (subject, relation, object) triples from data/cancer/ into Neo4j.

Uses larger chunks than the embedding pipeline (store_index.py) — relation
extraction benefits from more surrounding context than a 500-char retrieval
chunk gives it. Run this after data/cancer/ changes; MERGE makes it safe to
re-run without duplicating nodes or relationships.
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.graph_extraction import build_extractor, extract_triples
from src.graph_store import get_driver, write_triples

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "data/cancer/")
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
# Lower concurrency + explicit retries: at 6 workers on a larger corpus we hit
# OpenAI's 200k-TPM rate limit repeatedly, and those chunks were silently
# dropped rather than retried — a real, easy-to-miss gap in graph coverage.
MAX_WORKERS = 3


def load_chunks():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        for piece in splitter.split_text(text):
            chunks.append((piece, source))
    return chunks


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunk(s) from {DATA_DIR} for extraction.")

    # max_retries: the OpenAI client retries 429s with backoff internally, but
    # the default (2) isn't enough headroom under sustained concurrent load.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=8)
    extractor = build_extractor(llm)

    all_triples = []
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(extract_triples, extractor, text, source): source
            for text, source in chunks
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                triples = future.result()
                all_triples.extend(triples)
            except Exception as exc:  # noqa: BLE001 - one bad chunk shouldn't kill the run
                errors += 1
                print(f"  [{done}/{len(chunks)}] extraction error: {exc}")
            if done % 20 == 0 or done == len(chunks):
                print(f"  [{done}/{len(chunks)}] chunks processed, {len(all_triples)} triple(s) so far")

    print(f"Extraction done: {len(all_triples)} triple(s), {errors} chunk error(s).")

    driver = get_driver()
    try:
        print("Writing triples into Neo4j ...")
        write_triples(driver, all_triples)
        print("Done.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
