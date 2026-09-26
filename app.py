import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH", "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
)

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

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
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    response = rag_chain.invoke({"input": user_input})
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
