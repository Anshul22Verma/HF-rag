import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import json
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_openai import OpenAI
from langchain.schema import BaseRetriever
from typing import List
import torch

# Constants (Replace with your actual values)
VECTOR_DB_FILE = "/Users/vermaa/Downloads/HF_rag.index"
EMBEDDING_MODEL = "all-mpnet-base-v2"
DATA_FILE = "/Users/vermaa/Downloads/HF_rag_data.json"
EF_SEARCH = 40
OPENAI_KEY = "***"

st.set_page_config(
    page_title="HF/HTx",  # Change the tab title here
    page_icon="🫀",  # Change the tab icon here (use an emoji or URL)
)

prompt_template = """
Use the following context to answer the user's question. Only use the context do not hallucinate or generate new information.
Context: {context}
Question: {question}
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Load FAISS index and data (run only once)
@st.cache_resource
def load_resources():
    index = faiss.read_index(VECTOR_DB_FILE)
    index.hnsw.efSearch = EF_SEARCH

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    data = data["documents"]

    all_texts = []
    all_metadatas = []

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    for document in data:
        document_id = document["document_id"]
        title = document["title"]
        sections = document["text"]

        for section_title, section_data in sections.items():
            section_subsections = section_data["title"]
            section_text = section_data["text"]
            total_text = section_subsections + " : " + section_text

            all_texts.append(section_text)
            all_metadatas.append({
                "document_id": document_id,
                "title": title,
                "section_title": section_title
            })

    return index, all_texts, all_metadatas, embedding_model

index, all_texts, all_metadatas, embedding_model = load_resources()

class FAISSRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = embedding_model.encode([query]).astype('float32')
        distances, indices = index.search(query_embedding, k=8)
        documents = []
        for i in indices[0]:
            documents.append(Document(page_content=all_texts[i], metadata=all_metadatas[i]))
        return documents

# Load LLM (run only once)
@st.cache_resource
def load_llm():
    return OpenAI(openai_api_key=OPENAI_KEY)

llm = load_llm()
retriever = FAISSRetriever()


# Streamlit UI
st.title("Heart Failure (UHN), Chatbot")
# Styling
st.markdown(
    """
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .user {
        background-color: #e6f7ff;
    }
    .assistant {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            result = qa_chain.invoke(prompt)
            full_response = result["result"]
            source_documents = result["source_documents"]

            print("Prompt sent to LLM:")
            print(qa_chain.combine_documents_chain.llm_chain.prompt.format(context='\n'.join([doc.page_content for doc in source_documents]), question=prompt))
        except Exception as e:
            full_response = f"An error occurred: {e}"

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
