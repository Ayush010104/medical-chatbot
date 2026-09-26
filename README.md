# Medical Chatbot

A retrieval-augmented Q&A chatbot over your own medical PDFs, built with LangChain,
Pinecone (vector store), a local Llama-2 model (via CTransformers), and a Flask chat UI.

## How it works

1. PDFs in `data/` are loaded, chunked, and embedded with a sentence-transformers model.
2. The embeddings are upserted into a Pinecone index.
3. At query time, the chatbot retrieves the most relevant chunks from Pinecone and
   feeds them to a local Llama-2 model as context for its answer.

## Setup

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in `.env` with:

- `PINECONE_API_KEY` — from your [Pinecone](https://www.pinecone.io/) account (free tier works).
- `PINECONE_INDEX_NAME` — any name you like; the index is created automatically if missing.
- `LLAMA_MODEL_PATH` — path to the local Llama-2 GGML model file (see next step).

### 3. Download the Llama-2 model

Download a quantized Llama-2-7B-Chat GGML model (e.g. from TheBloke's Hugging Face
repo) and place it at `model/llama-2-7b-chat.ggmlv3.q4_0.bin` (or update
`LLAMA_MODEL_PATH` in `.env` to wherever you put it).

### 4. Add your source PDFs

Drop any medical reference PDFs into `data/`.

### 5. Build the Pinecone index

```bash
python store_index.py
```

Re-run this any time you add or change PDFs in `data/`.

### 6. Run the chatbot

```bash
python app.py
```

Open http://localhost:8080 in your browser.

## Project structure

```
├── app.py              # Flask app + RAG chain
├── store_index.py      # One-off script to (re)build the Pinecone index
├── src/
│   ├── helper.py        # PDF loading, chunking, embeddings
│   └── prompt.py        # System prompt for the LLM
├── templates/chat.html  # Chat UI
├── static/style.css     # Chat UI styling
├── data/                # Put your source PDFs here
└── model/               # Put the downloaded Llama-2 GGML file here
```

## Disclaimer

This chatbot is for informational purposes only and is not a substitute for
professional medical advice, diagnosis, or treatment.
