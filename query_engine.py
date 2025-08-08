import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

# Config
INDEX_DIR = 'index'
EMBED_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

TOP_K = 15          # candidates from FAISS
TOP_N = 3           # final reranked results
COS_SIM_THRESHOLD = 0.35
DEBUG = True

# Load embedding model
model = SentenceTransformer(EMBED_MODEL_NAME)

# Load reranker
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)

# Load FAISS index
index = faiss.read_index(f"{INDEX_DIR}/docs.index")

# Load metadata
sources = []
chunks = []
with open(f"{INDEX_DIR}/sources.txt", "r", encoding='utf-8') as f:
    for line in f:
        src, text = line.strip().split("|||", 1)
        sources.append(src)
        chunks.append(text)

# 🔹 Ollama Mistral call
def ollama_generate(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR calling Ollama Mistral: {e}]"

# 🔁 Reranking
def rerank(query, candidates):
    pairs = [(query, item["text"]) for item in candidates]
    inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1).numpy()

    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, _ in reranked[:TOP_N]]

# 🧠 Generate answer from Mistral via Ollama
def generate_answer(query, context_chunks):
    context_text = "\n".join([f"- {chunk}" for chunk in context_chunks])
    prompt = f"""You are a helpful assistant. Use the provided context to answer the question.

Context:
{context_text}

Question:
{query}

Answer concisely and factually based on the context. If the answer is not in the context, say so."""
    return ollama_generate(prompt)

# 🔍 Query
def query(q, top_k=TOP_K):
    query_embedding = model.encode([q], normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding), top_k)

    candidates = []
    for idx, score in zip(I[0], D[0]):
        if score >= COS_SIM_THRESHOLD:
            candidates.append({
                "source": sources[idx],
                "text": chunks[idx],
                "score": score
            })

    if DEBUG:
        print(f"\n[DEBUG] Retrieved {len(candidates)} candidates:")
        for c in candidates:
            print(f"  - Score: {c['score']:.4f} | Source: {c['source']}")

    reranked_results = rerank(q, candidates)

    if DEBUG:
        print(f"\n[DEBUG] Top {TOP_N} after reranking:")
        for r in reranked_results:
            print(f"  - {r['source']} | {r['text'][:80]}...")

    # Extract only the text chunks for Mistral
    context_chunks = [r["text"] for r in reranked_results]
    llm_answer = generate_answer(q, context_chunks)

    return {
        "answer": llm_answer,
        "sources": reranked_results
    }

# CLI Testing
if __name__ == "__main__":
    while True:
        q = input("🔍 Enter your question (or 'exit'): ")
        if q.lower() == 'exit':
            break

        output = query(q)
        print("\n💡 Answer:")
        print(output["answer"])
        print("\n📌 Supporting Sources:")
        for i, res in enumerate(output["sources"], 1):
            print(f"Result #{i} | Source: {res['source']}")
            print(res['text'])
            print("-" * 50)

__all__ = ["query", "sources"]
