import os
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configuration
DATA_DIR = 'data'
INDEX_DIR = 'index'
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
CHUNK_SIZE = 100
OVERLAP = 20

# Load sentence transformer model
model = SentenceTransformer(MODEL_NAME)

def read_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.strip().split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest():
    texts = []
    sources = []

    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)

        if fname.endswith('.pdf'):
            text = read_pdf(fpath)
        elif fname.endswith('.docx'):
            text = read_docx(fpath)
        else:
            print(f"⚠️ Skipping unsupported file type: {fname}")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"⚠️ Skipping empty or unreadable file: {fname}")
            continue

        texts.extend(chunks)
        sources.extend([fname] * len(chunks))

    if not texts:
        print("❌ No valid text chunks found. Exiting ingestion.")
        return

    print(f"✅ Total chunks to embed: {len(texts)}")

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save index and metadata
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "sources.txt"), "w", encoding='utf-8') as f:
        for s, t in zip(sources, texts):
            f.write(f"{s}|||{t.strip()}\n")

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest()
