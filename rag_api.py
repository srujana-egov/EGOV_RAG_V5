# Load environment variables FIRST
import os
from dotenv import load_dotenv
load_dotenv()  # This must be before any other imports

# Now import other modules
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any
import logging
import json
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug: Print current working directory and .env file location
cwd = os.getcwd()
env_path = os.path.join(cwd, '.env')
logger.info(f"Current working directory: {cwd}")
logger.info(f"Looking for .env file at: {env_path}")

# Debug: List files in current directory
logger.info("Files in current directory:")
for f in os.listdir(cwd):
    if f.startswith('.'):  # Show hidden files
        logger.info(f"  {f}")

# Debug: Check if .env exists and can be read
if os.path.exists(env_path):
    logger.info(".env file found, contents:")
    try:
        with open(env_path, 'r') as f:
            # Mask sensitive values
            lines = []
            for line in f:
                if 'key' in line.lower() or 'secret' in line.lower() or 'password' in line.lower() or 'api' in line.lower():
                    key, sep, value = line.partition('=')
                    lines.append(f"{key}{sep}***MASKED***")
                else:
                    lines.append(line.strip())
            logger.info("\n".join(lines))
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
else:
    logger.warning(".env file not found!")

# Debug: Print all environment variables (masking sensitive ones)
logger.info("Environment variables:")
for key, value in os.environ.items():
    if any(s in key.lower() for s in ['key', 'secret', 'password', 'api']):
        logger.info(f"  {key}=***MASKED***")
    else:
        logger.info(f"  {key}={value}")

# Now import your RAG functions
try:
    from retrieval import hybrid_retrieve_pg
    from generator import generate_rag_answer
except ImportError as e:
    logger.error(f"Failed to import RAG modules: {e}")
    sys.exit(1)

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
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Check if OPENAI_API_KEY is set
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        logger.error("Please make sure your .env file exists and contains OPENAI_API_KEY")
        sys.exit(1)
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
