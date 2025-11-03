# Load environment variables FIRST
import os
import sys
import logging
from dotenv import load_dotenv
load_dotenv()  # This must be before any other imports

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now import other modules
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2 import pool
import json

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection pool
db_pool = None

try:
    # Create a connection pool
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        user=os.getenv('PGUSER', 'neondb_owner'),
        password=os.getenv('PGPASSWORD', 'npg_UfakhcQ9MdL5'),
        host=os.getenv('PGHOST', 'ep-gentle-queen-a1gqdwzu.ap-southeast-1.aws.neon.tech'),
        port=os.getenv('PGPORT', '5432'),
        database=os.getenv('PGDATABASE', 'neondb'),
        sslmode=os.getenv('PGSSLMODE', 'require')
    )
    logger.info("Successfully created database connection pool")
except Exception as e:
    logger.error(f"Error creating database connection pool: {e}")
    sys.exit(1)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# In rag_api.py, update the query_rag function:

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Import here to avoid circular imports
        from retrieval import hybrid_retrieve_pg
        from generator import generate_rag_answer
        
        # Call hybrid_retrieve_pg without the conn parameter
        docs_and_meta = hybrid_retrieve_pg(
            query=request.query, 
            top_k=request.top_k
        )
        
        answer = generate_rag_answer(request.query, lambda q, top_k: docs_and_meta)
        
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
    finally:
        # Always return the connection to the pool
        if conn:
            db_pool.putconn(conn)

if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please make sure your .env file exists and contains all required variables")
        sys.exit(1)
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
