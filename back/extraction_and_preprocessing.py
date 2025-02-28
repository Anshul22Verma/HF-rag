import fitz
import os
import re
import json
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from llama_cpp import Llama  # Import Llama from llama-cpp-python
from langchain.prompts import PromptTemplate

# Configuration (Adjust paths and model settings)
PDF_DIRECTORY = "/Users/vermaa/Downloads/HF_documents"
OUTPUT_JSON_FILE = "/Users/vermaa/Downloads/HF_rag_data.json"
# LOCAL_MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"  # Path to your local Llama model
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # HF model ID
HF_HUB_TOKEN= "****"  # token to download the model
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Sentence Transformer model, using a small model for quick embedding
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Initialize local model
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, token=HF_HUB_TOKEN)  # Token here!
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_ID,
    token=HF_HUB_TOKEN, 
    device_map="auto",
    # load_in_4bit=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.1,
)

llm = HuggingFacePipeline(pipeline=pipe)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF. (PyPDF2 is the other option)
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")  # Or "dict" if you need structured data
        doc.close()  # VERY IMPORTANT: Close the document!
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None


def clean_text(text):
    # LLM-powered cleaning (example prompt - customize as needed)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Clean and normalize this medical text, removing tables, figures, captions, 
        and any other non-textual or irrelevant elements.  Focus on preserving
        the core medical information and making the text suitable for a 
        Retrieval Augmented Generation (RAG) system.  

        Specifically:
        * Remove or replace table content with a summary.
        * Remove figure captions and any text clearly associated with figures.
        * Remove or simplify complex formatting.
        * Standardize medical terminology where appropriate.
        * Correct any OCR errors if present.
        * Ensure the text is well-formatted and readable.

        Text:
        {text}
        """
    )
    cleaned_text = llm.invoke(prompt.format(text=text))

    # Basic regex cleaning (after LLM) - often still needed
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra whitespace
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip() # Remove extra newlines

    # Example 2: Remove superscript citations ^4,5
    cleaned_text = re.sub(r"\^(?:\d+(?:,\s*\d+)*)", "", cleaned_text).strip() # Improved Regex
    return cleaned_text


def extract_chapters(text):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract the chapter titles from the following text.  
        Return each chapter title on a separate line.  Prioritize 
        consistency.  If the text uses a specific format (e.g., "Chapter 1:", 
        "Chapter 2:"), adhere to that format.  If the format varies, try to 
        standardize it to the most common pattern.

        Text:
        {text}
        """
    )
    chapter_titles = llm.invoke(prompt.format(text=text))
    return chapter_titles.split('\n')  # Refine splitting as needed


def extract_chapter_text(text, chapter_title):  # Defined here!
    """Extracts the text content of a chapter from the book text."""

    # 1. Regex-Based Extraction (If Chapters Are Clearly Delimited) - Example
    # match = re.search(rf"(Chapter \d+:\s*{chapter_title}).*?(?=(Chapter \d+|$))", text, re.DOTALL | re.IGNORECASE)  # Case-insensitive
    # if match:
    #     return match.group(0).strip()  # Remove leading/trailing whitespace

    # 2. LLM-Based Extraction (More Robust - Use if regex fails or inconsistent)
    prompt = PromptTemplate(
        input_variables=["text", "chapter_title"],
        template="""
        Extract the text content of chapter '{chapter_title}' from this book text.
        If the chapter title is not found, return an empty string.

        Text:
        {text}
        """
    )
    chapter_text = llm.invoke(prompt.format(text=text, chapter_title=chapter_title))
    return chapter_text.strip() # Remove leading/trailing whitespace


def chunk_text(text, chunk_size=400, overlap=50): # Token-based chunking
    """
    Tokenize and chunk text into smaller pieces for processing.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks


def create_metadata(pdf_file, chapter, page_num, audience="Patient"):
    return {
        "source": pdf_file,
        "chapter": chapter,
        "page_number": page_num, # You'll need to implement page number extraction
        "audience": audience  # "Patient" or "Clinician"
    }


def process_pdf(pdf_path, audience: str="Patient"):
    """
    Process a PDF book, extracting text, cleaning it, and chunking it into smaller pieces.
    """
    text = extract_text_from_pdf(pdf_path)
    if text is None:
        return []

    # chunking the text because the model can't handle very large inputs like books
    chunks = chunk_text(text, chunk_size=100) # Chunk the raw extracted text *first*

    all_chunks = []
    page_num = 1 # Placeholder

    for chunk in tqdm(chunks, total=len(chunks), desc=f"Processing {pdf_path}"):
        cleaned_chunk = clean_text(chunk)  # Clean each chunk
        chapters_in_chunk = extract_chapters(cleaned_chunk) # Extract chapters from the chunk

        for chapter in chapters_in_chunk:
            chapter_text = extract_chapter_text(cleaned_chunk, chapter) # Extract the chapter text

            if chapter_text: # Check if chapter_text was extracted successfully
                sub_chunks = chunk_text(chapter_text) # Sub-chunk the chapter text if needed
                for sub_chunk in sub_chunks:
                    embedding = embedding_model.encode(sub_chunk)
                    metadata = create_metadata(os.path.basename(pdf_path), chapter, page_num, audience)
                    all_chunks.append({
                        "text_chunk": sub_chunk,
                        "embedding": embedding.tolist(),
                        "metadata": metadata
                    })
                    page_num += 1 # Placeholder
    return all_chunks


if __name__ == "__main__":
    # Main processing loop
    all_data = []
    for filename in tqdm(os.listdir(PDF_DIRECTORY), total=len(os.listdir(PDF_DIRECTORY))):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            # Determine audience (you'll need a way to do this - maybe filename-based?)
            audience = "Patient"  # Or "Doctor" - make this dynamic!
            all_data.extend(process_pdf(pdf_path, audience))

    # Save to JSON
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(all_data, f, indent=4)

    print(f"Processed data saved to {OUTPUT_JSON_FILE}")
