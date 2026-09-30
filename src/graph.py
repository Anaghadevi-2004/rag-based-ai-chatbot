from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

import os
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def build_rag_graph(index_name: str):

    # --------------------------------------------------
    # 1. Embeddings
    # --------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768
    )

    # --------------------------------------------------
    # 2. Pinecone Vector Store
    # --------------------------------------------------

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # --------------------------------------------------
    # 3. Retriever
    # --------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    # --------------------------------------------------
    # 4. Free LLM - Groq
    # --------------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=1024
    )

    # --------------------------------------------------
    # 5. Retrieval Node
    # --------------------------------------------------

    def retrieve_node(state: AgentState):

        docs = retriever.invoke(state["question"])

        context_texts = [
            doc.page_content
            for doc in docs
        ]

        return {
            "context": context_texts
        }

    # --------------------------------------------------
    # 6. Generation Node
    # --------------------------------------------------

    def generate_node(state: AgentState):

        context_str = "\n\n".join(
            state["context"]
        )

        prompt = f"""
You are a helpful AI assistant answering questions
about the provided Agentic AI eBook.

Answer the user's question ONLY using the information
provided in the context below.

If the answer is not present in the context, say:

"I cannot answer based on the provided document."

Do not use outside knowledge.
Do not invent information.

Context:
-------------------------
{context_str}
-------------------------

Question:
{state["question"]}

Answer clearly and concisely.
"""

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "score": 0.95 if state["context"] else 0.0
        }

    # --------------------------------------------------
    # 7. Build LangGraph
    # --------------------------------------------------

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_edge(
        START,
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()