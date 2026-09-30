from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
import time
from dotenv import load_dotenv

load_dotenv()

def run_ingestion(pdf_path: str, index_name: str):
    print("Loading document...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    print("Connecting to Pinecone and preparing embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768
    )
    
    # Initialize the vector store connection
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Upload in batches of 80 to stay safely under Google's 100 requests/minute limit
    batch_size = 80
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Uploading batch {i // batch_size + 1} (Chunks {i + 1} to {i + len(batch)})...")
        vector_store.add_documents(batch)
        
        # If there are more chunks remaining, pause for 60 seconds
        if i + batch_size < len(chunks):
            print("Batch successful! Waiting 60 seconds to reset Google's free tier rate limit...")
            time.sleep(60)
            
    print(f"Successfully ingested all {len(chunks)} chunks into Pinecone index '{index_name}'.")
    return vector_store

if __name__ == "__main__":
    target_index = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    run_ingestion("data/Ebook-Agentic-AI.pdf", target_index)