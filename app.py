import os
import fitz  # PyMuPDF
import gradio as gr
import requests
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from pathlib import Path


# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")




env_path = Path(__file__).parent / ".env"
print("Looking for .env at:", env_path)

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENROUTER_API_KEY")
print("Loaded API Key:", api_key)


# Check if key loaded
if not api_key:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file!")

# Load embedding model

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
stored_chunks = []

# --- Chunking PDF ---
def chunk_pdf_text(file_path, chunk_size=500):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --- Store Embeddings ---
def store_chunks(chunks):
    embeddings = embed_model.encode(chunks)
    index.add(embeddings)
    stored_chunks.extend(chunks)

# --- Search ---
def retrieve_relevant_chunks(query, k=5):
    query_embed = embed_model.encode([query])
    D, I = index.search(query_embed, k)
    return [stored_chunks[i] for i in I[0]]

# --- Call OpenRouter LLM ---
def generate_answer(query, context):
    prompt = f"""Use the following context to answer the question.
Context:
{context}

Question: {query}
Answer:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://openrouter.ai",  # optional but useful
        "X-Title": "academic-rag-assistant"
    }

    payload = {
        "model": "qwen/qwen-2.5-7b-instruct",  # you can change model here  (mistralai/mistral-7b-instruct)
        "messages": [
            {"role": "system", "content": "You are a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=payload)

    try:
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(response.text)}"

# --- Main Gradio Function ---
def process_file_and_query(pdf_file, query):
    try:
        chunks = chunk_pdf_text(pdf_file.name)
        store_chunks(chunks)
        relevant_chunks = retrieve_relevant_chunks(query)
        context = " ".join(relevant_chunks)
        return generate_answer(query, context)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- Gradio UI ---
gr.Interface(
    fn=process_file_and_query,
    inputs=[
        gr.File(label="📄 Upload PDF", type="filepath"),
        gr.Textbox(label="💬 Ask a question")
    ],
    outputs=gr.Textbox(label="Answer", lines=15),
    title="Academic RAG Assistant",
    description="Upload a research paper PDF and ask a question based only on its content. Powered by OpenRouter LLM + RAG."
).launch(share=True)
