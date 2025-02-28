import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

# Configuration
DATA_FILE = "/Users/vermaa/Downloads/HF_rag_data.json"
VECTOR_DB_FILE = "/Users/vermaa/Downloads/HF_rag.index"

M = 16
EF_CONSTRUCTION = 500
EMBEDDING_MODEL = "all-mpnet-base-v2"

def create_faiss_index():
    # Load data
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    
    data = data["documents"]
    # Create embeddings, texts, and metadatas
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    all_embeddings = []
    all_texts = []
    all_metadatas = []

    for document in data:
        document_id = document["document_id"]
        title = document["title"]
        sections = document["text"]

        for section_title, section_data in tqdm(sections.items(), total=len(sections), desc=f"Processing document: {title}"):
            section_subsections = section_data["title"]
            section_text = section_data["text"]
            total_text = section_subsections + " : " + section_text
            embedding = embedding_model.encode(total_text).astype('float32')

            all_embeddings.append(embedding)
            all_texts.append(section_text)
            all_metadatas.append({
                "document_id": document_id,
                "title": title,
                "section_title": section_title
            })

    # Convert embeddings to numpy array
    embeddings_np = np.array(all_embeddings)

    # Create FAISS index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(embeddings_np)

    # Save the FAISS index
    faiss.write_index(index, VECTOR_DB_FILE)
    print(f"FAISS index saved to {VECTOR_DB_FILE}")


if __name__ == "__main__":
    if not os.path.exists(VECTOR_DB_FILE):
        create_faiss_index()
    else:
        print(f"FAISS index already exists at {VECTOR_DB_FILE}")