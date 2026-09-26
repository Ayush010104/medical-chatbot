import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.safety import EMERGENCY_RESPONSE, LOW_CONFIDENCE_RESPONSE, is_red_flag

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH", "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
)
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "3"))
# Pinecone cosine similarity is in [-1, 1]; below this, the retrieved context
# is too weak to ground an answer in, so we refuse rather than let the LLM
# generate something fluent but unsupported.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

llm = CTransformers(
    model=LLAMA_MODEL_PATH,
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.4},
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"].strip()

    # 1. Escalation: deterministic, runs before any retrieval or generation.
    if is_red_flag(user_input):
        return jsonify(
            {"answer": EMERGENCY_RESPONSE, "sources": [], "confidence": "emergency"}
        )

    # 2. Confidence gate: only generate if we actually found grounding for it.
    results = docsearch.similarity_search_with_score(user_input, k=RETRIEVAL_K)

    if not results or results[0][1] < CONFIDENCE_THRESHOLD:
        return jsonify(
            {"answer": LOW_CONFIDENCE_RESPONSE, "sources": [], "confidence": "low"}
        )

    # 3. Citations: return the exact chunks the answer is grounded in.
    docs = [doc for doc, _score in results]
    answer = question_answer_chain.invoke({"input": user_input, "context": docs})

    sources = [
        {
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "page": doc.metadata.get("page"),
            "excerpt": doc.page_content[:200].strip(),
            "score": round(float(score), 3),
        }
        for doc, score in results
    ]

    return jsonify({"answer": answer, "sources": sources, "confidence": "normal"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
