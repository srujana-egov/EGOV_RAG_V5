from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import List, Dict, Any
import logging
import json

# Import your existing RAG functions
from retrieval import hybrid_retrieve_pg
from generator import generate_rag_answer

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    top_k: int = 5

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Use the same retrieval and generation logic as your Streamlit app
        docs_and_meta = hybrid_retrieve_pg(request.query, top_k=request.top_k)
        answer = generate_rag_answer(request.query, lambda q, top_k: docs_and_meta)
        
        # Format the response
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc,
                    "metadata": meta
                } 
                for doc, meta in docs_and_meta
            ]
        }
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Make sure required environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        exit(1)
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
