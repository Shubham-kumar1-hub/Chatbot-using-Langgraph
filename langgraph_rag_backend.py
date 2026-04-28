from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# PDF retriever store
# -------------------

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        # ---------- CLEAN TEXT ----------
        texts = []
        metadatas = []

        for doc in chunks:
            text = doc.page_content

            if text is None:
                continue

            # Ensure pure unicode string for tokenizer compatibility
            text = str(text).strip()
            # Remove null bytes and non-UTF-8 characters that break the tokenizer
            text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            # Strip control characters (except newlines and tabs) that cause TextEncodeInput errors
            text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
            text = text.strip()

            if len(text) == 0:
                continue

            texts.append(text)
            metadatas.append(doc.metadata)

        if len(texts) == 0:
            raise ValueError("No valid text extracted from the PDF")

        # ---------- VECTOR STORE ----------
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(texts),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(texts),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# Tools
# -------------------

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]

llm_with_tools = llm.bind_tools(tools)

# -------------------
# State
# -------------------


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# Nodes
# -------------------


def chat_node(state: ChatState, config=None):

    thread_id = None

    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "For questions about uploaded PDFs call rag_tool "
            f"and include thread_id `{thread_id}`. "
            "You can also use web search, stock price, and calculator tools."
        )
    )

    messages = [system_message, *state["messages"]]

    response = llm_with_tools.invoke(messages, config=config)

    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# Checkpointer
# -------------------

conn = sqlite3.connect("chatbot.db", check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

# -------------------
# Graph
# -------------------

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# Helpers
# -------------------


def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:

    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:

    return _THREAD_METADATA.get(str(thread_id), {})