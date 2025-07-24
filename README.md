```markdown
# Gemini-Powered RAG Chatbot using LangChain

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using:
- Google's **Gemini 1.5 Flash** (for chat and embeddings),
- **FAISS** (for vector search),
- **LangChain** (for orchestration),
- and **LangSmith** (for tracing/debugging).

---

##  Project Overview

The RAG pipeline has two main components:

###  1. Indexing Pipeline

Used to preprocess and store documents in a searchable format.

**Steps:**
```

Load --> Split --> Embed --> Store

```

- **Load**: Load documents using `Document` loader
- **Split**: (optional) Use text splitters for large texts
- **Embed**: Use Gemini embeddings (`embedding-001`)
- **Store**: Store vectors in FAISS for fast retrieval

---

###  2. Retrieval + Generation

Used at inference time to fetch relevant content and generate answers.

**Flow:**
```

Question --> Retrieve --> Prompt --> LLM --> Answer

````

- **Retrieve**: Pull top-k relevant chunks from FAISS
- **Prompt**: Combine question + retrieved content
- **LLM**: Use Gemini 1.5 Flash to generate the answer
- **Answer**: Return final response to user

---

## How to Run

###  Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
````

###  .env File

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_google_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project_name   # optional
```

---

###  Run the App

```bash
python rag_app.py
```

You'll be prompted to ask a question, and the system will return a generated answer along with the retrieved source documents.

---

##  FAISS Index

The FAISS vector index is saved locally as:

```
my_faiss_index/
```

You can reuse it later without re-indexing documents.

---

##  LangSmith Integration

The code includes optional **LangSmith** support:

* Enables tracing and monitoring your LLM pipeline.
* Helps debug prompt flow and understand model behavior.

📌 Setup your [LangSmith account](https://smith.langchain.com/) to get the `LANGCHAIN_API_KEY`.

---

##  Google Gemini API Billing

Be cautious about API usage:

* Google offers **free tier** via AI Studio.
* Use **budget alerts** in your Google Cloud Console.
* Monitor usage and avoid unnecessary charges.

[🔗 Gemini Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)

---

##  Tech Stack

| Tool         | Purpose                    |
| ------------ | -------------------------- |
| Gemini Flash | Fast LLM for generation    |
| FAISS        | Vector store for retrieval |
| LangChain    | Orchestrates RAG pipeline  |
| LangSmith    | Logs/traces LLM executions |
| dotenv       | Loads API keys from `.env` |

---

## Future Improvements

* Add PDF/Text file loaders
* Integrate a UI (e.g., Streamlit or Gradio)
* Add evaluation metrics with LangSmith
* Implement text chunking and metadata

---

## Example Query

```
Ask a question: What is RAG?

 Answer: RAG stands for Retrieval-Augmented Generation...

Sources:
[1] RAG stands for Retrieval-Augmented Generation.
[2] LangChain is a framework...
```

---

## References
https://python.langchain.com/docs/tutorials/rag/

