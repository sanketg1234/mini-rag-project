import os
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. API Setup ---
app = FastAPI()

# Allow our React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you'd lock this down to your frontend's URL
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

# --- 2. Build the RAG Index on Startup ---
print("Initializing RAG Pipeline...")
loader = DirectoryLoader("data/", glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""])
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("RAG Pipeline Ready!")

OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY" # Put your key here!

# --- 3. The Chat Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Fetch relevant chunks
        retrieved_docs = retriever.invoke(request.query)
        
        # Format the context to send back to the frontend AND to the LLM
        contexts = []
        context_text_for_llm = ""
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            contexts.append({"source": source, "content": content})
            context_text_for_llm += f"\nSource: {source}\nContent: {content}\n"

        # Create the prompt
      # Create the prompt
        prompt = f"""You are an AI assistant for a construction marketplace.
        Answer the user's question using ONLY the context provided below.
        You are allowed to synthesize the provided bullet points to directly answer the query. 
        If the context does not contain relevant information to address the core of the user's question, do not guess. Say "I do not have enough information to answer that based on the provided documents."

        Context:
        {context_text_for_llm}

        Question: {request.query}
        Answer:"""

        # Call the LLM
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
            # Return BOTH the answer and the context arrays to satisfy the UI transparency requirement
            return {"answer": answer, "context": contexts}
        else:
            raise HTTPException(status_code=500, detail="Error communicating with LLM")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this server from the terminal:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)