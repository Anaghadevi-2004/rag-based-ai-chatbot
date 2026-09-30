# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) based AI chatbot that answers questions strictly using information retrieved from an Agentic AI eBook.

The project uses **Python, LangGraph, Pinecone, Google Gemini Embeddings, Groq LLM, and FastAPI** to build an end-to-end document question-answering system.

---

## Project Overview

This project implements a complete RAG pipeline that:

1. Loads an Agentic AI PDF document.
2. Extracts text from the document.
3. Splits the document into smaller chunks.
4. Generates vector embeddings for the chunks using Google Gemini.
5. Stores the embeddings in Pinecone.
6. Retrieves the most relevant chunks for a user query.
7. Uses a Groq-hosted LLM to generate an answer based only on the retrieved context.
8. Exposes the RAG pipeline through a FastAPI `/chat` endpoint.
9. Returns the generated answer, retrieved context chunks, and a confidence score.
10. Includes benchmark queries for testing document grounding.

---

# 🏗️ Architecture

```text
                    ┌───────────────────────┐
                    │   Agentic AI PDF      │
                    │       eBook           │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    PyPDFLoader        │
                    │   Text Extraction     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ RecursiveCharacter    │
                    │    Text Splitter      │
                    │                       │
                    │ Chunk Size: 1000      │
                    │ Overlap: 200          │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Google Gemini         │
                    │ Embeddings             │
                    │ gemini-embedding-001 │
                    │ Dimension: 768        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │       Pinecone        │
                    │    Vector Database    │
                    │                       │
                    │   agentic-ai-index    │
                    └───────────┬───────────┘
                                │
                         User Question
                                │
                                ▼
                    ┌───────────────────────┐
                    │      FastAPI          │
                    │      /chat             │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      Retriever        │
                    │      Top K = 5        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      LangGraph        │
                    │                       │
                    │ Retrieve → Generate   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │       Groq LLM        │
                    │   GPT-OSS 20B         │
                    │                       │
                    │ Strict Context        │
                    │ Grounded Generation   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      API Response     │
                    │                       │
                    │ • Final Answer        │
                    │ • Retrieved Chunks    │
                    │ • Confidence Score    │
                    └───────────────────────┘

## Key Features
📄 PDF document ingestion
✂️ Recursive text chunking
🧠 Google Gemini embeddings
🔎 Semantic similarity search
🗄️ Pinecone vector database
🔄 LangGraph workflow orchestration
🤖 Groq LLM generation
🔒 Context-grounded responses
🚫 Protection against unsupported questions
⚡ FastAPI REST API
🧪 Benchmark testing
📊 Retrieved context and confidence score in API response

## Technology Stack
Technology	Purpose
Python	Core programming language
LangChain	RAG components and integrations
LangGraph	Stateful workflow orchestration
Google Gemini	Text embeddings
Pinecone	Vector database
Groq	LLM inference
FastAPI	REST API
Uvicorn	ASGI server
PyPDF	PDF text extraction
Streamlit	Optional UI
python-dotenv	Environment variable management
## 📂 Project Structure
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── graph.py
│   └── config.py
│
├── app.py
├── requirements.txt
├── .env
├── .env.example
├── README.md
└── tests_sample_queries.py
📄 Document

The knowledge source used by this project is an Agentic AI eBook.

The chatbot is designed to answer questions based on this document rather than using general web knowledge.

The PDF should be placed at:

data/Ebook-Agentic-AI.pdf
⚙️ Prerequisites

Before running the project, install:

Python 3.10+
Git
Pinecone account
Google AI API key
Groq API key

No OpenAI API key is required for the current implementation.

## Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project directory:

cd rag-agentic-ai
2. Create a virtual environment
Windows
python -m venv venv

Activate it:

venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies

Run:

pip install -r requirements.txt

The main dependencies include:

langchain
langgraph
langchain-community
langchain-pinecone
langchain-text-splitters
langchain-google-genai
langchain-groq
pinecone-client
pypdf
fastapi
uvicorn
streamlit
python-dotenv
Environment Configuration

Create a .env file in the project root.

GOOGLE_API_KEY=your_google_api_key

GROQ_API_KEY=your_groq_api_key

PINECONE_API_KEY=your_pinecone_api_key

PINECONE_INDEX_NAME=agentic-ai-index
Important

Never commit your .env file to GitHub.

Add the following to .gitignore:

.env
venv/
__pycache__/
*.pyc

A safe .env.example file can be included in the repository:

GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
Pinecone Configuration

Create a Pinecone index with:

Index Name: agentic-ai-index
Dimension: 768
Metric: cosine

The dimension must match the Gemini embedding configuration:

GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768
)
Document Ingestion

The ingestion pipeline is implemented in:

src/ingestion.py

The pipeline performs the following operations:

PDF
 ↓
PyPDFLoader
 ↓
Document Pages
 ↓
RecursiveCharacterTextSplitter
 ↓
Text Chunks
 ↓
Gemini Embeddings
 ↓
Pinecone
Chunking Configuration

The project uses:

chunk_size=1000
chunk_overlap=200

This creates overlapping chunks to preserve contextual information between neighboring sections.

Run Document Ingestion

Make sure the PDF exists at:

data/Ebook-Agentic-AI.pdf

Then run:

python src/ingestion.py

The ingestion process:

Loads the PDF.
Extracts document pages.
Splits the text into chunks.
Generates Gemini embeddings.
Uploads vectors to Pinecone.

For larger documents, batches are used during ingestion to help manage API rate limits.

🧠 RAG Workflow

The RAG workflow is implemented using LangGraph.

The graph contains two main nodes:

START
  │
  ▼
Retrieve
  │
  ▼
Generate
  │
  ▼
 END
1. Retrieve Node

The retrieve node queries Pinecone using the user's question.

The retriever is configured as:

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

Therefore, the top 5 relevant document chunks are retrieved.

2. Generate Node

The retrieved chunks are combined into a context.

The LLM receives:

Context
+
User Question

The prompt explicitly instructs the model to use only the provided document context.

The system also instructs the model not to invent information or use outside knowledge.

If the required information is not available, the expected response is:

I cannot answer based on the provided document.
🔒 Strict Grounding

One of the main goals of the project is preventing the chatbot from answering questions using unrelated external knowledge.

The generation prompt follows this principle:

Answer the user's question ONLY using the information
provided in the context.

Do not use outside knowledge.

Do not invent information.

If the answer is not present in the context, say:

"I cannot answer based on the provided document."

This helps ensure that responses remain grounded in the retrieved eBook content.

🌐 FastAPI API

The application exposes a REST API using FastAPI.

Start the API using:

uvicorn app:app --reload

The API will be available at:

http://127.0.0.1:8000
📌 API Endpoint
POST /chat
Request
{
    "query": "What is Agentic AI according to the eBook?"
}
Response
{
    "query": "What is Agentic AI according to the eBook?",
    "final_answer": "Agentic AI refers to AI systems that can operate autonomously toward goals, make decisions, plan actions, and execute tasks.",
    "retrieved_context_chunks": [
        "Relevant document chunk 1...",
        "Relevant document chunk 2...",
        "Relevant document chunk 3..."
    ],
    "confidence_score": 0.95
}
🧪 Testing

The project includes:

tests_sample_queries.py

Run:

python tests_sample_queries.py

The benchmark includes questions such as:

1. What is Agentic AI according to the eBook?

2. How do AI agents differ from traditional automation systems?

3. What are the core components of an Agentic Architecture?

4. What role does memory play in Agentic AI workflows?

5. Who won the 2022 FIFA World Cup?

The first four questions are related to the knowledge source.

The fifth question is an out-of-document validation test.

The chatbot should not answer the fifth question using general knowledge.

Expected behavior:

I cannot answer based on the provided document.
📊 Retrieval Testing

The retrieval system can also be tested independently.

For example, a retrieval test can return:

Retrieved documents: 5

Each retrieved document contains:

Document content
Source
Page number
Document metadata

Example:

DOCUMENT 1

Decision-Making Layer:
At the core of agentic systems is the decision-making layer...

This confirms that the Pinecone retrieval layer is successfully returning relevant document chunks.

📸 Screenshots
1. Project Structure

Add a screenshot showing the complete project structure.

Save the image inside:

screenshots/project-structure.png

Then add:

![Project Structure](screenshots/project-structure.png)
2. Pinecone Index

Add a screenshot of the Pinecone index showing:

Index name
Dimension
Vector count
Status

Save it as:

screenshots/pinecone-index.png

Add:

![Pinecone Index](screenshots/pinecone-index.png)
3. Document Ingestion

Add a screenshot of the terminal showing successful ingestion.

Example:

Loading document...
Splitting text into chunks...
Total chunks created: XX
Connecting to Pinecone and preparing embeddings...
Uploading batch 1...
...
Successfully ingested all XX chunks.

Save it as:

screenshots/ingestion.png

Add:

![Document Ingestion](screenshots/ingestion.png)
4. FastAPI Server

Add a screenshot showing:

uvicorn app:app --reload

with:

Uvicorn running on http://127.0.0.1:8000
Application startup complete.

Save it as:

screenshots/fastapi-server.png

Add:

![FastAPI Server](screenshots/fastapi-server.png)
5. API Documentation

FastAPI automatically provides interactive API documentation.

Open:

http://127.0.0.1:8000/docs

Take a screenshot showing:

POST /chat

Save it as:

screenshots/api-docs.png

Add:

![FastAPI Documentation](screenshots/api-docs.png)
6. Successful RAG Response

Use the FastAPI /docs interface or your test script to demonstrate a question such as:

What is Agentic AI according to the eBook?

The response should show:

final_answer
retrieved_context_chunks
confidence_score

Save the screenshot as:

screenshots/rag-response.png

Add:

![RAG Response](screenshots/rag-response.png)
7. Out-of-Document Query

Take a screenshot showing the system handling:

Who won the 2022 FIFA World Cup?

The system should respond that the answer cannot be provided based on the document.

Save as:

screenshots/grounding-test.png

Add:

![Grounding Test](screenshots/grounding-test.png)
📈 Example Testing Output

Example terminal output:

--- Query 1: What is Agentic AI according to the eBook? ---

FinalAnswer: Agentic AI refers to AI systems that can operate
autonomously toward goals, make decisions, plan actions, and
execute tasks using information from their environment.

Retrieved Context Chunks: 5
Confidence Score: 0.95

Another example:

--- Query 3: What are the core components of an Agentic Architecture? ---

FinalAnswer: The architecture includes components such as
decision-making, planning, learning, and execution layers.

Retrieved Context Chunks: 5
Confidence Score: 0.95

Out-of-document test:

--- Query 5: Who won the 2022 FIFA World Cup? ---

FinalAnswer: I cannot answer based on the provided document.

Retrieved Context Chunks: 5
Confidence Score: 0.95

Note: The confidence value currently represents a simple heuristic based on successful context retrieval, rather than a calibrated probability.

🔄 Complete Data Flow

The complete system operates as follows:

                    USER
                     │
                     ▼
              ┌─────────────┐
              │   FastAPI   │
              │   /chat     │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  LangGraph  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  Retrieve   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  Pinecone   │
              │ Vector Store│
              └──────┬──────┘
                     │
               Top 5 Chunks
                     │
                     ▼
              ┌─────────────┐
              │  Generate   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │    Groq     │
              │ GPT-OSS 20B │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   Answer    │
              │ + Context   │
              │ + Score     │
              └─────────────┘
🧩 LangGraph State

The application maintains the following state:

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float
State Fields
Field	Description
question	User's input question
context	Retrieved document chunks
answer	Generated response
score	Retrieval/context confidence score
🧠 Embedding Configuration

The project uses:

GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768
)

Therefore, the Pinecone index uses:

Dimension = 768

and:

Metric = cosine

The same embedding configuration is used during both ingestion and retrieval to ensure vector compatibility.

🤖 LLM Configuration

The generation layer uses Groq:

ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1024
)

A temperature of 0 is used to make responses more deterministic.

🔐 Security

API keys are stored using environment variables.

Example:

GOOGLE_API_KEY=your_key
GROQ_API_KEY=your_key
PINECONE_API_KEY=your_key

The .env file should never be committed to the repository.

The .gitignore file includes:

.env
venv/
__pycache__/
*.pyc
🧪 Validation Strategy

The application is validated using two types of questions.

In-domain questions

Questions whose answers are expected to exist in the Agentic AI eBook.

Examples:

What is Agentic AI?

What are the components of an Agentic Architecture?

How does Agentic AI differ from traditional automation?

What role does memory play in Agentic AI?
Out-of-domain question

A question unrelated to the document:

Who won the 2022 FIFA World Cup?

This test verifies that the chatbot does not simply answer using the LLM's general knowledge.

⚠️ Current Limitations
The confidence score is currently a simple heuristic based on whether context was retrieved.
Pinecone similarity retrieval can return chunks even when the query is unrelated to the document.
A production implementation could improve this by applying a similarity-score threshold or a dedicated relevance evaluation step.
The current system is optimized for a single knowledge source.
🔮 Future Improvements

Potential improvements include:

Similarity score thresholding
Cross-encoder reranking
More advanced confidence scoring
Conversational memory
Streaming responses
Source/page citations
Multiple document support
Metadata filtering
Hybrid keyword + semantic retrieval
Streamlit chat interface
Evaluation using RAGAS or similar evaluation frameworks
Query rewriting
Retrieval fallback strategies
📋 Requirements

The project uses the following dependencies:

langchain>=0.2.0
langgraph>=0.1.0
langchain-community>=0.2.0
langchain-pinecone>=0.1.0
langchain-text-splitters>=0.2.0
langchain-google-genai>=2.0.0
langchain-groq>=0.1.0
pinecone-client>=3.0.0
pypdf>=4.0.0
fastapi>=0.110.0
uvicorn>=0.28.0
streamlit>=1.32.0
python-dotenv>=1.0.0
▶️ Quick Start
# Clone repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

# Enter project
cd rag-agentic-ai

# Create virtual environment
python -m venv venv

# Windows activation
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure .env

# Ingest document
python src/ingestion.py

# Start FastAPI
uvicorn app:app --reload

# Run benchmark tests
python tests_sample_queries.py
```
