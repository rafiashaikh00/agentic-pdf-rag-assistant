import os
import sqlite3
from typing import Annotated, Dict, Any

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    START,
)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0,
)


# ============================================================
# 3. HUGGING FACE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ============================================================
# 4. CHROMA CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"


# ============================================================
# 5. THREAD-WISE RETRIEVERS
# ============================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}

_THREAD_METADATA: Dict[str, dict] = {}


# ============================================================
# 6. PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: str
):
    """
    Process uploaded PDF:
    PDF -> Documents -> Chunks -> Embeddings -> ChromaDB
    """

    temp_path = f"temp_{thread_id}.pdf"

    try:

        # ----------------------------------------------------
        # Save uploaded PDF temporarily
        # ----------------------------------------------------

        with open(temp_path, "wb") as f:
            f.write(file_bytes)


        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(temp_path)

        documents = loader.load()


        # ----------------------------------------------------
        # Split PDF into chunks
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(
            documents
        )


        # ----------------------------------------------------
        # Create unique collection for this thread
        # ----------------------------------------------------

        collection_name = (
            "pdf_" +
            thread_id.replace("-", "_")
        )


        # ----------------------------------------------------
        # Create Chroma vector store
        # ----------------------------------------------------

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )


        # ----------------------------------------------------
        # Add chunks to Chroma
        # ----------------------------------------------------

        vector_store.add_documents(
            chunks
        )


        # ----------------------------------------------------
        # Create retriever
        # ----------------------------------------------------

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4
            }
        )


        # ----------------------------------------------------
        # Store retriever for current thread
        # ----------------------------------------------------

        _THREAD_RETRIEVERS[thread_id] = retriever


        # ----------------------------------------------------
        # Store document metadata
        # ----------------------------------------------------

        _THREAD_METADATA[thread_id] = {
            "filename": filename,
            "pages": len(documents),
            "chunks": len(chunks),
            "collection": collection_name
        }


        return {
            "filename": filename,
            "pages": len(documents),
            "chunks": len(chunks)
        }


    finally:

        # ----------------------------------------------------
        # Delete temporary PDF
        # ----------------------------------------------------

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 7. WEB SEARCH TOOL
# ============================================================

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ============================================================
# 8. CALCULATOR TOOL
# ============================================================

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> float:
    """
    Perform basic mathematical calculations.
    """

    if operation == "add":

        return first_num + second_num


    elif operation == "subtract":

        return first_num - second_num


    elif operation == "multiply":

        return first_num * second_num


    elif operation == "divide":

        if second_num == 0:
            return "Error: Cannot divide by zero."

        return first_num / second_num


    else:

        return (
            "Invalid operation. "
            "Use add, subtract, multiply or divide."
        )


# ============================================================
# 9. RAG TOOL
# ============================================================

@tool
def rag_tool(
    query: str,
    thread_id: str
) -> str:
    """
    Search the PDF uploaded in the current conversation.
    """

    retriever = _THREAD_RETRIEVERS.get(
        thread_id
    )


    if retriever is None:

        return (
            "No PDF is uploaded in this conversation. "
            "Please upload a PDF first."
        )


    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    documents = retriever.invoke(
        query
    )


    if not documents:

        return (
            "No relevant information was found "
            "in the uploaded PDF."
        )


    results = []


    for doc in documents:

        # PyPDFLoader uses zero-based page number
        page_number = doc.metadata.get(
            "page"
        )

        if page_number is not None:
            page_number += 1


        text = doc.page_content.strip()


        results.append(
            f"Page: {page_number}\n"
            f"Content: {text}"
        )


    return "\n\n---\n\n".join(
        results
    )


# ============================================================
# 10. TOOLS
# ============================================================

tools = [
    search_tool,
    calculator,
    rag_tool,
]


# ============================================================
# 11. BIND TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 12. GRAPH STATE
# ============================================================

class ChatState:

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 13. CHAT NODE
# ============================================================

def chat_node(
    state: ChatState,
    config: RunnableConfig
):

    # Get current conversation thread
    thread_id = (
        config["configurable"]["thread_id"]
    )


    # --------------------------------------------------------
    # System instructions
    # --------------------------------------------------------

    system_message = SystemMessage(
        content=f"""
You are a helpful AI assistant.

Current thread ID:
{thread_id}

Available tools:

1. calculator
   Use for mathematical calculations.

2. web search
   Use when the user asks for current
   or web-based information.

3. rag_tool
   Use when the user asks about
   information contained in the uploaded PDF.

Important rules:

- For PDF questions, use rag_tool.
- Pass this exact thread_id to rag_tool:
  {thread_id}
- Do not invent PDF information.
- If the answer is not present in the PDF,
  say that it was not found.
- When answering from the PDF,
  mention the relevant page number.
"""
    )


    messages = [
        system_message
    ] + state["messages"]


    response = llm_with_tools.invoke(
        messages
    )


    return {
        "messages": [response]
    }


# ============================================================
# 14. TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# 15. SQLITE CHECKPOINTER
# ============================================================

connection = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(
    connection
)


# ============================================================
# 16. BUILD GRAPH
# ============================================================

graph = StateGraph(
    ChatState
)


# Nodes
graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)


# ============================================================
# 17. EDGES
# ============================================================

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# 18. COMPILE
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 19. RETRIEVE ALL THREADS
# ============================================================

def retrieve_all_threads():

    threads = set()


    for checkpoint in checkpointer.list(
        None
    ):

        thread_id = (
            checkpoint.config
            .get("configurable", {})
            .get("thread_id")
        )


        if thread_id:

            threads.add(
                thread_id
            )


    return list(threads)


# ============================================================
# 20. CHECK DOCUMENT
# ============================================================

def thread_has_document(
    thread_id: str
) -> bool:

    return (
        thread_id in _THREAD_RETRIEVERS
    )


# ============================================================
# 21. DOCUMENT METADATA
# ============================================================

def thread_document_metadata(
    thread_id: str
):

    return _THREAD_METADATA.get(
        thread_id
    )