import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Configuration
VECTOR_DB_FILE = "HF_rag.index"
EMBEDDING_MODEL = "all-mpnet-base-v2"
DATA_FILE = "data.json"

# Load FAISS index
index = faiss.read_index(VECTOR_DB_FILE)
dimension = index.d  # The dimension of the embeddings

# Load the new document data (document_id 5)
with open(DATA_FILE, "r") as f:
    data = json.load(f)

new_document = None
# already added document with chuncked video transcription and image captions in data.json as document_id=5
for document in data["documents"]:
    if document["document_id"] == 5:
        new_document = document
        break

# Prepare the model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Extract embeddings for the new document (document 5)
new_embeddings = []
new_texts = []
new_metadatas = []

# Extract and embed sections of document 5
for section_title, section_data in new_document["text"].items():
    section_subsections = section_data["title"]
    section_text = section_data["text"]
    total_text = section_subsections + " : " + section_text
    embedding = embedding_model.encode(total_text).astype('float32')

    # Add the embedding to the list
    new_embeddings.append(embedding)
    new_texts.append(section_text)
    new_metadatas.append({
        "document_id": new_document["document_id"],
        "title": new_document["title"],
        "section_title": section_title
    })

# Convert new embeddings to numpy array
new_embeddings_np = np.array(new_embeddings)

# Add the new embeddings to the index
index.add(new_embeddings_np)

# Save the updated FAISS index
faiss.write_index(index, VECTOR_DB_FILE)
print(f"FAISS index updated with document 5 and saved to {VECTOR_DB_FILE}")
