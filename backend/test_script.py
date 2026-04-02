import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import json

print("Loading documents...")
loader = DirectoryLoader(
    "data/", 
    glob="**/*.md", 
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

print("Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Building the FAISS vector index...")
vector_store = FAISS.from_documents(chunks, embedding_model)
print("Vector index built successfully!")

vector_store.save_local("faiss_index")
print("✅ Saved FAISS index locally to the 'faiss_index' folder!")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

test_query = "What factors affect construction project delays?"
print(f"\n--- Testing Retrieval ---")
print(f"Query: '{test_query}'")

retrieved_docs = retriever.invoke(test_query)

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n[Retrieved Chunk {i}]")
    print(f"Source Document: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content: {doc.page_content}")


print("\n--- Generating Answer with LLM ---")

OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY" 

context_text = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in retrieved_docs])

prompt = f"""You are an AI assistant for a construction marketplace.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, do not guess. Say "I do not have enough information to answer that based on the provided documents."

Context:
{context_text}

Question: {test_query}
Answer:"""

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {"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"}",
        "Content-Type": "application/json"
    },
    data=json.dumps({
        "model": "openrouter/free",
        "messages": [{"role": "user", "content": prompt}]
    })
)

if response.status_code == 200:
    answer = response.json()['choices'][0]['message']['content']
    print("\n=========================================")
    print("[Final Generated Answer]")
    print("=========================================")
    print(answer)
    print("\n(Successfully met transparency requirement: Context and Answer displayed)")
else:
    print(f"Error calling LLM: {response.text}")    