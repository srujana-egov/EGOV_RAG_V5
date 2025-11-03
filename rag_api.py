from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generator import generate_rag_answer
from retrieval import hybrid_retrieve_pg
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        # Call your existing RAG functions
        retrieved_chunks = hybrid_retrieve_pg(request.query, top_k=request.top_k)
        answer = generate_rag_answer(request.query, retrieved_chunks)
        
        return {
            "answer": answer,
            "sources": [{"content": chunk.page_content, "metadata": chunk.metadata} 
                       for chunk in retrieved_chunks]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
