# Construction Assistant AI (Mini RAG Pipeline)

This project is a full-stack Retrieval-Augmented Generation (RAG) application built for a construction marketplace. It features a Python/FastAPI backend and a React frontend.

## 🚀 How to Run Locally
1. **Clone the repository:** `git clone <your-repo-link>`
2. **Backend Setup:**
   - Navigate to the root directory.
   - Install dependencies: `pip install fastapi uvicorn pydantic langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu requests`
   - Add your OpenRouter API key to `main.py`.
   - Start the server: `python main.py`
3. **Frontend Setup:**
   - Open a new terminal and navigate to the `frontend` folder.
   - Install dependencies: `npm install`
   - Start the React app: `npm run dev`

## 🧠 Architecture & Technical Choices

### 1. Embedding Model & LLM Selection
* **Embedding Model:** I used `all-MiniLM-L6-v2` via HuggingFace `sentence-transformers`. Why? It is an open-source, highly efficient, and fast model that runs perfectly locally without needing an external API key.
* **LLM:** I utilized the `openrouter/free` auto-routing endpoint. Why? It fulfills the assignment's free OpenRouter tier requirement while automatically handling model rotation, ensuring the API doesn't fail if a specific free model goes offline.

### 2. Document Chunking and Retrieval
* **Chunking Logic:** Documents are loaded using LangChain's `DirectoryLoader` and split using the `RecursiveCharacterTextSplitter`. I chose a `chunk_size` of 500 characters with a `chunk_overlap` of 50 characters. This ensures the chunks are large enough to retain meaningful context but small enough to keep the LLM prompt focused.
* **Retrieval:** I implemented a local vector store using FAISS. For each user query, the system performs a semantic similarity search and retrieves the top 3 (`k=3`) most relevant document chunks.

### 3. Enforcing Grounding (Preventing Hallucinations)
Grounding is strictly enforced through Prompt Engineering. The LLM is wrapped in an instructional prompt that injects the retrieved FAISS chunks and explicitly commands the model to:
1. Answer the question using ONLY the provided context.
2. Say "I do not have enough information..." if the context does not contain the answer.
This completely prevents the model from relying on its general training data.

## 📊 Quality Analysis (Optional Bonus)
*Note: Include a brief summary here if you decide to test 8-15 questions and evaluate the system's hallucinations and retrieval accuracy!*