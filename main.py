# main.py

from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF
import numpy as np

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load PDF and extract text
doc = fitz.open("sample.pdf")  # 📝 Replace with your actual PDF path
texts = [page.get_text() for page in doc]
full_text = "\n".join(texts)

# Split into chunks (simple split for now)
chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

# Embed chunks
embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Ask a question
query=str(input("Please ask any question: "))
#query = "What is well baby well mother about?"
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding), k=3)

# Show top 3 matching chunks
print("\nTop Answers:")
for i in I[0]:
    print(f"- {chunks[i][:300]}...\n")
