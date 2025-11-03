# At the top with other imports
import psycopg2
from psycopg2 import pool

# ... (keep all existing imports and setup code) ...

# After loading environment variables, add database connection pool
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

# Update your query_rag function to use the connection pool
@app.post("/query")
async def query_rag(request: QueryRequest):
    conn = None
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Get a connection from the pool
        conn = db_pool.getconn()
        
        # Use the connection with your retrieval function
        docs_and_meta = hybrid_retrieve_pg(
            request.query, 
            top_k=request.top_k,
            conn=conn  # Make sure hybrid_retrieve_pg accepts a conn parameter
        )
        
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
    finally:
        # Always return the connection to the pool
        if conn:
            db_pool.putconn(conn)

# At the bottom of the file
if __name__ == "__main__":
    # Check for required environment variables
    required_vars = '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'
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
