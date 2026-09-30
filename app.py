from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_rag_graph
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Agentic AI RAG API")

target_index = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
graph = build_rag_graph(index_name=target_index)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        initial_state = {
            "question": request.query, 
            "context": [], 
            "answer": "", 
            "score": 0.0
        }
        
        result = graph.invoke(initial_state)

        return QueryResponse(
            query=request.query,
            final_answer=result.get("answer", "I cannot answer based on the provided document."),
            retrieved_context_chunks=result.get("context", []),
            confidence_score=result.get("score", 0.0)
        )
    except Exception as e:
        print(f"Error executing query: {e}")
        return QueryResponse(
            query=request.query,
            final_answer="I cannot answer based on the provided document due to an internal error.",
            retrieved_context_chunks=[],
            confidence_score=0.0
        )
