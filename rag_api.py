from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Database configuration
DB_CONFIG = {
    "user": "neondb_owner",
    "password": "npg_UfakhcQ9MdL5",
    "host": "ep-gentle-queen-a1gqdwzu.ap-southeast-1.aws.neon.tech",
    "port": "5432",
    "database": "neondb",
    "sslmode": "require"
}

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(autocommit=True)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

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
    conn = None
    try:
        logger.info(f"Received query: {request.query}")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Example query - modify according to your schema
        cursor.execute("""
            SELECT * FROM your_table 
            WHERE your_search_column ILIKE %s 
            LIMIT %s
        """, (f"%{request.query}%", request.top_k))
        
        results = cursor.fetchall()
        
        if not results:
            return {
                "answer": f"No results found for: {request.query}",
                "sources": []
            }
            
        return {
            "answer": f"Found {len(results)} results for: {request.query}",
            "sources": [
                {
                    "content": str(result),  # Convert result row to string
                    "metadata": {"source": "neondb"}
                } 
                for result in results
            ]
        }
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # First install required package if not already installed
    try:
        import psycopg2
    except ImportError:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
