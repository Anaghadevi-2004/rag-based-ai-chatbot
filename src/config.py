import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 768
LLM_MODEL = "gemini-2.5-flash"