import requests
import json

BASE_URL = "http://localhost:8000/chat"

TEST_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?"
]

def run_benchmarks():
    for idx, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n--- Query {idx}: {query} ---")
        try:
            response = requests.post(BASE_URL, json={"query": query})
            response.raise_for_status()
            data = response.json()
            
            print(f"FinalAnswer: {data.get('final_answer')}")
            print(f"Retrieved Context Chunks: {len(data.get('retrieved_context_chunks', []))}")
            print(f"Confidence Score: {data.get('confidence_score')}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error executing test: {e}")

if __name__ == "__main__":
    run_benchmarks()