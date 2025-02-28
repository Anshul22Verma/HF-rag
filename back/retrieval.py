import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import Document
import json

# Configuration
VECTOR_DB_FILE = "/Users/vermaa/Downloads/HF_rag.index"
EMBEDDING_MODEL = "all-mpnet-base-v2"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
HF_HUB_TOKEN = "***"
DATA_FILE = "/Users/vermaa/Downloads/HF_rag_data.json"

EF_SEARCH = 100 # # HNSW parameter, controls search speed/accuracy


# Load data
with open(DATA_FILE, "r") as f:
    data = json.load(f)

texts = [item["text_chunk"] for item in data]
metadatas = [item["metadata"] for item in data]

# Load FAISS index
index = faiss.read_index(VECTOR_DB_FILE)
index.hnsw.efSearch = EF_SEARCH

# Create retrieval functions
embedding_model = SentenceTransformer(EMBEDDING_MODEL) # Embedding model initialized

def faiss_retriever(query, k=4):
    """
     FAISS uses the brute-force k-nearest neighbors (k-NN) search algorithm with Euclidean (L2) distance
    """
    query_embedding = embedding_model.encode([query]).astype('float32') # Embedding model now used.
    distances, indices = index.search(query_embedding, k)
    results = []
    for i in indices[0]:
        results.append({
            "page_content": texts[i],
            "metadata": metadatas[i]
        })
    return results

def faiss_retriever_langchain(query, k=4):
    results = faiss_retriever(query, k)
    documents = [Document(page_content=result["page_content"], metadata=result["metadata"]) for result in results]
    return documents

# Initialize LLM
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, use_auth_token=HF_HUB_TOKEN)
model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_ID, use_auth_token=HF_HUB_TOKEN)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=faiss_retriever_langchain)

# Query
query = "What are the treatments for type 2 diabetes?"
result = qa_chain.invoke(query)
print(result["result"])