from query_engine import query
from evaluation_data import evaluation_sets as evaluation_pairs
from sentence_transformers import SentenceTransformer, util  # NEW
import os

# Load the same model used in ingestion
similarity_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def search_index(question, filename, top_k=10):
    results = query(question, top_k=top_k * 2)  # Fetch more than needed

    # Filter by source
    filtered = [res for res in results if os.path.basename(res["source"]) == filename]
    if not filtered:
        return ""

    # Re-rank based on semantic similarity
    question_emb = similarity_model.encode(question, convert_to_tensor=True)
    scored = []
    for res in filtered:
        chunk_emb = similarity_model.encode(res["text"], convert_to_tensor=True)
        score = util.cos_sim(question_emb, chunk_emb).item()
        scored.append((score, res["text"]))

    # Sort by score and take top_k
    scored.sort(reverse=True)
    top_chunks = [text for _, text in scored[:top_k]]

    return " ".join(top_chunks)

def evaluate():
    correct = 0
    total = 0

    for doc, pairs in evaluation_pairs.items():
        print(f"\n🔍 Evaluating questions for: {doc}")
        for pair in pairs:
            question = pair["question"]
            expected_answer = pair["answer"]
            retrieved_answer = search_index(question, doc)

            print(f"\nQ: {question}")
            print(f"✅ Expected: {expected_answer}")
            print(f"📄 Retrieved: {retrieved_answer}")

            # Semantic similarity comparison
            expected_emb = similarity_model.encode(expected_answer, convert_to_tensor=True)
            retrieved_emb = similarity_model.encode(retrieved_answer, convert_to_tensor=True)
            cos_sim = util.cos_sim(expected_emb, retrieved_emb).item()

            if cos_sim >= 0.6:
                correct += 1
            else:
                print(f"❌ Incorrect (Similarity: {cos_sim:.2f})")

            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n🎯 Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")

if __name__ == "__main__":
    evaluate()
