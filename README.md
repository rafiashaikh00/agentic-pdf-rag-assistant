# 🤖 Agentic PDF RAG Assistant

An agentic AI assistant built with **LangGraph** that combines **PDF-based Retrieval-Augmented Generation (RAG)**, tool calling, web search, calculator capabilities, persistent conversations, and streaming responses in a Streamlit application.

The system can answer questions from uploaded PDF documents, use external tools when required, and maintain separate conversation threads using LangGraph checkpointing.

---

## ✨ Features

* 📄 **PDF Upload & Processing**
* 🔎 **Retrieval-Augmented Generation (RAG)**
* 🧠 **LangGraph Agent Workflow**
* 🔧 **LLM Tool Calling**
* 🌐 **Web Search**
* 🧮 **Calculator Tool**
* 🗃️ **ChromaDB Vector Store**
* 🤗 **Hugging Face Embeddings**
* 💬 **Thread-Based Conversations**
* 💾 **SQLite Conversation Persistence**
* ⚡ **Streaming AI Responses**
* 📑 **PDF Page References**
* 🖥️ **Streamlit Interface**
* 🔐 **Environment-based API Key Management**

---

## 🏗️ Architecture

The application uses **LangGraph** to coordinate the LLM, tools, document retrieval, and conversation state.

```text
                              ┌──────────────┐
                              │     User     │
                              └──────┬───────┘
                                     │
                                     ▼
                           ┌──────────────────┐
                           │  Streamlit UI    │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │    LangGraph     │
                           │   Chat Workflow  │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │     Groq LLM     │
                           └────────┬─────────┘
                                    │
                         ┌──────────┼──────────┐
                         │          │          │
                         ▼          ▼          ▼
                    ┌─────────┐ ┌─────────┐ ┌────────────┐
                    │ PDF RAG │ │  Web    │ │ Calculator │
                    │  Tool   │ │ Search  │ │    Tool    │
                    └────┬────┘ └─────────┘ └────────────┘
                         │
                         ▼
                    ┌─────────┐
                    │ChromaDB │
                    └────┬────┘
                         │
                         ▼
                ┌──────────────────┐
                │Hugging Face      │
                │Embeddings        │
                └──────────────────┘

                         │
                         ▼
                ┌──────────────────┐
                │  Final Response  │
                └──────────────────┘


              Conversation Persistence
                         │
                         ▼
                ┌──────────────────┐
                │ SQLite Checkpoint│
                │    Storage       │
                └──────────────────┘
```

---

## 🔄 How It Works

### 1. Upload a PDF

The user uploads a PDF through the Streamlit interface.

### 2. Document Processing

The PDF is loaded using `PyPDFLoader` and divided into smaller chunks using a recursive text splitter.

### 3. Generate Embeddings

Each document chunk is converted into a vector representation using a Hugging Face sentence-transformer embedding model.

### 4. Store Vectors

The generated embeddings are stored in **ChromaDB**, allowing relevant document chunks to be retrieved later.

### 5. User Query

The user asks a question through the chat interface.

### 6. LangGraph Workflow

LangGraph manages the conversation flow and allows the LLM to determine when a tool is required.

### 7. Tool Calling

The assistant can use different tools depending on the query:

| Tool          | Purpose                                    |
| ------------- | ------------------------------------------ |
| 📄 PDF RAG    | Retrieve information from the uploaded PDF |
| 🌐 Web Search | Retrieve current/web-based information     |
| 🧮 Calculator | Perform mathematical calculations          |

### 8. Retrieval

For PDF-related questions, the RAG tool searches ChromaDB for the most relevant document chunks.

### 9. Response Generation

The retrieved information and conversation context are provided to the LLM, which generates the final response.

### 10. Conversation Persistence

LangGraph uses a **SQLite checkpointer** to maintain conversation state for individual thread IDs.

### 11. Streaming

The generated response is streamed to the Streamlit interface rather than waiting for the entire response before displaying it.

---

## 🧠 Agent Workflow

The core LangGraph workflow follows this pattern:

```text
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Chat Node   │
                    │   LLM       │
                    └──────┬──────┘
                           │
                    Tool required?
                       /        \
                     Yes         No
                     /            \
                    ▼              ▼
             ┌─────────────┐   ┌─────────┐
             │ Tool Node   │   │   END   │
             └──────┬──────┘   └─────────┘
                    │
                    ▼
             ┌─────────────┐
             │ Chat Node   │
             │   LLM       │
             └──────┬──────┘
                    │
                    ▼
                  END
```

This allows the model to:

1. Receive the user's request.
2. Decide whether a tool is required.
3. Execute the selected tool.
4. Receive the tool result.
5. Generate the final response.

---

## 📄 PDF RAG Pipeline

The document retrieval pipeline works as follows:

```text
PDF Upload
    │
    ▼
PyPDFLoader
    │
    ▼
Document Pages
    │
    ▼
Text Splitting
    │
    ▼
Document Chunks
    │
    ▼
Hugging Face Embeddings
    │
    ▼
ChromaDB
    │
    ▼
Similarity Retrieval
    │
    ▼
Relevant Context
    │
    ▼
Groq LLM
    │
    ▼
Final Answer + Page Reference
```

---

## 💾 Conversation Persistence

Each conversation is associated with a unique `thread_id`.

```text
User
 │
 ▼
Thread ID
 │
 ▼
LangGraph
 │
 ▼
SQLite Checkpointer
 │
 ▼
Conversation State
```

This allows the application to maintain separate conversations rather than mixing messages from different chats.

Users can create new conversations and switch between previous conversation threads through the Streamlit sidebar.

---

## 🛠️ Tech Stack

| Technology            | Role                                |
| --------------------- | ----------------------------------- |
| **Python**            | Application development             |
| **LangGraph**         | Agent workflow and state management |
| **LangChain**         | LLM and tool integration            |
| **Groq**              | LLM inference                       |
| **Hugging Face**      | Text embeddings                     |
| **ChromaDB**          | Vector database                     |
| **PyPDF**             | PDF processing                      |
| **DuckDuckGo Search** | Web search                          |
| **SQLite**            | Conversation checkpoint storage     |
| **Streamlit**         | User interface                      |

---

## 📂 Project Structure

```text
agentic-pdf-rag-assistant/
│
├── backend/
│   └── stream_rag_backend.py
│
├── frontend/
│   └── rag_frontend.py
│
├── screenshots/
│   ├── main-interface.png
│   ├── pdf-rag.png
│   └── conversations.png
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🖥️ Screenshots

### Main Interface

![Main Interface](screenshots/main-interface.png)

### PDF RAG

![PDF RAG](screenshots/pdf-rag.png)

### Conversation Management

![Conversation Management](screenshots/conversations.png)

---

## 🔐 Environment Variables

The application uses environment variables for API credentials.

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Never commit your `.env` file or API keys to the repository.**

A `.env.example` file is included to show the required environment variable without exposing credentials.

---

## ⚙️ Installation

Clone the repository and install the dependencies listed in `requirements.txt`.

The project requires Python and the packages specified in the dependency file.

After configuring the environment variables, start the Streamlit application.

---

## 🧪 Example Use Cases

### Ask questions about a PDF

```text
User:
What are the main findings discussed in the document?

Assistant:
[Retrieved information from the PDF]
Page: 4
...
```

### Mathematical calculation

```text
User:
Calculate 125 × 48.

Assistant:
6000
```

### Web-based question

```text
User:
Search the web for the latest information about X.

Assistant:
[Uses web search tool]
...
```

---

## 🔍 Engineering Highlights

* Graph-based agent workflow using LangGraph
* Multiple LLM tools integrated into a single workflow
* Retrieval-Augmented Generation over uploaded PDF documents
* Vector similarity search using ChromaDB
* Hugging Face sentence-transformer embeddings
* Thread-based conversation management
* SQLite-backed LangGraph checkpointing
* Streaming responses in Streamlit
* PDF page-aware responses
* Separation of frontend and backend responsibilities
* Environment-based secret management

---

## 🚀 Future Improvements

The current implementation provides the core agentic RAG workflow. Possible future improvements include:

* 🔌 MCP integration
* 🔎 Hybrid search
* 🎯 Retrieval reranking
* 📊 RAG evaluation
* 🔬 LangSmith observability
* 🌐 FastAPI backend
* 🔐 User authentication
* ☁️ Cloud deployment
* 📚 Multi-document knowledge bases
* 🗂️ Improved document lifecycle management

---

## 📌 Current Scope

This project currently focuses on:

**Agentic AI + Tool Calling + PDF RAG + Conversation Persistence + Streaming**


---

