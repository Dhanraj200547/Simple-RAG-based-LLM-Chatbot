import os
from dotenv import load_dotenv

load_dotenv()                # Load environment variables from .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")        # Get keys from environment
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY       # Set key for LangChain & Gemini
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings     # Langchain Gemini LLM and Embeddings

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_core.documents import Document       # Sample documents
docs = [
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="FAISS is a vector store used for similarity search."),
    Document(page_content="OpenAI offers powerful language models like GPT-4."),
    Document(page_content="RAG stands for Retrieval-Augmented Generation."),
]

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

embedding_dim = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(embedding_dim)        # Creating FAISS index

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(docs)        # Adding documents to vector store

retriever = vector_store.as_retriever(search_kwargs={"k": 3})       # setting up a retriever

from langchain.chains import RetrievalQA        # Setting up RAG RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = input("Ask a question: ")   # Running query
response = qa_chain.invoke(query)

print("\n📌 Answer:", response["result"])
print("\n📚 Sources:")
for i, doc in enumerate(response["source_documents"], start=1):
    print(f"\n[{i}] {doc.page_content}")

vector_store.save_local("my_faiss_index")   # Saving FAISS index locally
