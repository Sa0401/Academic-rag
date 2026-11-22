import os
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------------------------------------------
# Environment & Configuration Initialization
# -----------------------------------------------------------------------------
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "academic-rag").strip()

st.set_page_config(
    page_title="ScholarRAG Pro • Academic Intelligence Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Bespoke Dark Titanium UI Styling (Custom CSS)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f1f5f9;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* App Background & Canvas */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(17, 24, 39, 0.95) 0%, rgba(10, 15, 29, 1) 90%);
}

/* Top App Brand Bar */
.brand-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    backdrop-filter: blur(16px);
    margin-bottom: 24px;
}

.brand-title {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-badge {
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.35);
}

/* Glassmorphism Metric Tiles */
.stat-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
}

.stat-label {
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 700;
    margin-bottom: 2px;
}

.stat-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #f8fafc;
}

/* Prominent Question Command Box */
.question-panel {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px -4px rgba(99, 102, 241, 0.1);
}

/* Grounded Source Card */
.source-card {
    background: rgba(15, 23, 42, 0.75);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #cbd5e1;
}

.score-pill {
    background: rgba(52, 211, 153, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 600;
    float: right;
}

/* Custom Tabs */
div[data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 16px;
}

div[data-baseweb="tab"] {
    padding: 10px 20px;
    font-weight: 600;
    font-size: 0.90rem;
    border-radius: 10px 10px 0 0;
}

/* Hide Default Headers */
header[data-testid="stHeader"] {
    background: transparent;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Optional Dependencies Verification
# -----------------------------------------------------------------------------
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# -----------------------------------------------------------------------------
# Cached Embeddings & LLM Clients
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_embedding_model():
    """Loads SentenceTransformer with local offline files preference."""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            try:
                return SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                return None
    return None


def get_llm_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://scholar-rag.pro",
            "X-Title": "ScholarRAG Studio"
        }
    )


# -----------------------------------------------------------------------------
# Benchmark Sample Paper Data ("Attention Is All You Need" - Vaswani et al.)
# -----------------------------------------------------------------------------
BENCHMARK_PAPER_CHUNKS = [
    {
        "id": "chunk_1_0",
        "page": 1,
        "text": (
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
            "that include an encoder and a decoder. The Transformer eschews recurrence and convolutions, relying entirely "
            "on an attention mechanism to draw global dependencies between input and output. The Transformer allows for "
            "significantly more parallelization and can reach a new state of the art in translation quality after being "
            "trained for as little as twelve hours on eight P100 GPUs."
        )
    },
    {
        "id": "chunk_2_1",
        "page": 2,
        "text": (
            "The Transformer follows an encoder-decoder architecture. The encoder is composed of a stack of N = 6 identical "
            "layers. Each layer has two sub-layers: a multi-head self-attention mechanism, and a simple, position-wise "
            "fully connected feed-forward network. We employ residual connections around each of the two sub-layers, "
            "followed by layer normalization: LayerNorm(x + Sublayer(x)). The output of each sub-layer is of dimension d_model = 512."
        )
    },
    {
        "id": "chunk_3_2",
        "page": 3,
        "text": (
            "Scaled Dot-Product Attention: The input consists of queries and keys of dimension d_k, and values of dimension d_v. "
            "We compute the matrix of outputs as: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. The scale factor 1/sqrt(d_k) "
            "counteracts the dot products growing large in magnitude for large values of d_k, which would push the softmax "
            "function into regions that have extremely small gradients."
        )
    },
    {
        "id": "chunk_4_3",
        "page": 4,
        "text": (
            "Multi-Head Attention: Instead of performing a single attention function with d_model-dimensional queries, keys and values, "
            "we found it beneficial to linearly project the queries, keys and values h times with different learned projections to "
            "d_k, d_k and d_v dimensions. We compute MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O where "
            "head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V). In our work we employ h = 8 parallel attention heads. For each: "
            "d_k = d_v = d_model / h = 64."
        )
    },
    {
        "id": "chunk_5_4",
        "page": 5,
        "text": (
            "Positional Encoding: Since our model contains no recurrence and no convolution, in order for the model to make use "
            "of the order of the sequence, we must inject some information about the relative or absolute positions of the tokens. "
            "We use sine and cosine functions of different frequencies: PE_(pos, 2i) = sin(pos / 10000^(2i/d_model)) and "
            "PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))."
        )
    },
    {
        "id": "chunk_6_5",
        "page": 6,
        "text": (
            "Benchmark Results: On the WMT 2014 English-to-German translation task, the big Transformer model establishes a "
            "new state-of-the-art BLEU score of 28.4, outperforming existing best models, including ensembles, by over 2.0 BLEU. "
            "On the WMT 2014 English-to-French translation task, our model achieves a BLEU score of 41.8, outperforming all "
            "previously published single models at 1/4 the training cost."
        )
    }
]


# -----------------------------------------------------------------------------
# Multi-Tier PDF Extraction & Chunking (PyMuPDF4LLM + Direct Text + RapidOCR)
# -----------------------------------------------------------------------------
def extract_and_chunk_pdf_safe(
    file_bytes: bytes,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> Tuple[List[Dict[str, Any]], int, str]:
    """
    Extracts text page-by-page from PDF with a 3-tier fallback architecture:
    1. PyMuPDF4LLM structured markdown (ideal for academic tables & multi-column layouts)
    2. PyMuPDF text & block streams
    3. Neural RapidOCR for scanned image-only PDFs
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    if total_pages == 0:
        raise ValueError("The PDF document contains 0 pages.")

    pages_text: List[Tuple[int, str]] = []
    extraction_method = "direct"

    # Tier 1: Try PyMuPDF4LLM for rich structured markdown
    try:
        import pymupdf4llm
        md_pages = pymupdf4llm.to_markdown(doc, page_chunks=True)
        if md_pages and any(p.get("text", "").strip() for p in md_pages):
            for idx, p in enumerate(md_pages):
                txt = p.get("text", "").strip()
                if txt:
                    pages_text.append((idx + 1, txt))
            extraction_method = "academic-markdown"
    except Exception:
        pass

    # Tier 2: Standard text & block streams
    if not pages_text:
        for page_idx in range(total_pages):
            try:
                page_text = doc[page_idx].get_text("text").strip()
                if not page_text:
                    # Try block text
                    blocks = doc[page_idx].get_text("blocks")
                    page_text = "\n".join([b[4] for b in blocks if len(b) > 4 and b[4].strip()]).strip()
                if page_text:
                    pages_text.append((page_idx + 1, page_text))
            except Exception:
                continue
        extraction_method = "direct-text"

    # Tier 3: Neural OCR fallback for scanned image PDFs
    total_extracted_chars = sum(len(txt) for _, txt in pages_text)
    if total_extracted_chars < 50:
        pages_text = []
        try:
            from rapidocr_onnxruntime import RapidOCR
            import numpy as np
            ocr = RapidOCR()
            for page_idx in range(total_pages):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=150)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                if pix.n == 4:
                    import cv2
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                ocr_results, _ = ocr(img_np)
                if ocr_results:
                    recognized_text = " ".join([r[1] for r in ocr_results if len(r) > 1 and r[1].strip()])
                    if recognized_text.strip():
                        pages_text.append((page_idx + 1, recognized_text.strip()))
            if pages_text:
                extraction_method = "neural-ocr"
        except Exception as ocr_err:
            pass

    if not pages_text:
        raise ValueError(
            "Could not extract readable text from this PDF through direct streams or OCR. "
            "Please ensure the document contains legible text or clear scans."
        )

    chunks = []
    chunk_id = 0

    for page_num, text in pages_text:
        words = text.split()
        if not words:
            continue

        step = max(1, chunk_size - chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.strip()) > 25:
                chunks.append({
                    "id": f"chunk_{page_num}_{chunk_id}",
                    "text": chunk_text,
                    "page": page_num
                })
                chunk_id += 1

    if not chunks:
        raise ValueError("Document content was too sparse to generate contextual chunks.")

    return chunks, total_pages, extraction_method


# -----------------------------------------------------------------------------
# Vector Stores: Local In-Memory & Pinecone Serverless
# -----------------------------------------------------------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = {}


def upsert_local_vector_db(doc_id: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "page": chunk["page"],
            "embedding": emb
        })
    st.session_state.vector_db[doc_id] = records


def query_local_vector_db(doc_id: str, query_emb: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    import numpy as np
    records = st.session_state.vector_db.get(doc_id, [])
    if not records:
        return []

    q_vec = np.array(query_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return []

    results = []
    for r in records:
        emb_vec = np.array(r["embedding"], dtype=np.float32)
        denom = q_norm * np.linalg.norm(emb_vec)
        score = float(np.dot(q_vec, emb_vec) / denom) if denom != 0 else 0.0
        results.append({
            "id": r["id"],
            "text": r["text"],
            "page": r["page"],
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


class SafePineconeStore:
    def __init__(self, api_key: str, index_name: str = "academic-rag", dimension: int = 384):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone library is not available.")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self):
        try:
            if hasattr(self.pc, "has_index") and self.pc.has_index(self.index_name):
                return
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception:
            # If already exists or permission check passes, continue
            pass

    def upsert(self, doc_id: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        vectors = []
        for c, emb in zip(chunks, embeddings):
            vectors.append({
                "id": f"{doc_id}_{c['id']}",
                "values": emb,
                "metadata": {"text": c["text"], "page": c["page"], "doc_id": doc_id}
            })
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i : i + 100], namespace=doc_id)

    def search(self, doc_id: str, query_emb: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        res = self.index.query(vector=query_emb, namespace=doc_id, top_k=top_k, include_metadata=True)
        out = []
        for match in res.get("matches", []):
            out.append({
                "id": match["id"],
                "text": match["metadata"].get("text", ""),
                "page": match["metadata"].get("page", 1),
                "score": match.get("score", 0.0)
            })
        return out


# -----------------------------------------------------------------------------
# Structured LLM Synthesis & Diagram Generator
# -----------------------------------------------------------------------------
ACADEMIC_PROMPT = """You are a Principal AI Scientist analyzing academic research papers.
Answer the user's research question rigorously using the provided excerpts. Cite specific pages using [Page X].

CRITICAL: You MUST provide a clear visual diagram illustrating the concept, architecture, pipeline, or benchmark.
Respond strictly in JSON format:
{
  "answer": "Detailed academic explanation with clear Markdown formatting and citations like [Page 2]...",
  "visual_type": "graphviz" | "mermaid" | "table",
  "visual_title": "Descriptive title for the architecture or workflow diagram",
  "visual_code": "Valid Graphviz DOT code (e.g. 'digraph G { rankdir=TB; node [shape=box, style=rounded]; Input -> Attention -> Output; }') or Mermaid flowchart. Do NOT wrap inside backticks within this JSON field."
}
"""


def synthesize_with_visuals(
    query: str,
    chunks: List[Dict[str, Any]],
    llm_client: OpenAI,
    model_name: str
) -> Dict[str, Any]:
    context_str = "\n\n".join([f"[Excerpt from Page {c['page']}]:\n{c['text']}" for c in chunks])
    prompt = f"Context:\n{context_str}\n\nResearch Question: {query}"

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ACADEMIC_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.15
        )
        raw_text = response.choices[0].message.content
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            clean = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
            return json.loads(clean)
    except Exception as e:
        # Fallback response with programmatic architecture diagram
        return {
            "answer": f"**Synthesis Summary**: Based on the retrieved academic excerpts:\n\n{context_str[:500]}...\n\n*(Notice: LLM API returned: {str(e)})*",
            "visual_type": "graphviz",
            "visual_title": "Pipeline Architecture (Grounded Fallback)",
            "visual_code": "digraph G { rankdir=LR; node [shape=box, style=rounded, fontname=\"Helvetica\"]; Query -> VectorRetrieval -> ContextChunks -> LLMSynthesis; }"
        }


# -----------------------------------------------------------------------------
# Diagram Rendering Engine
# -----------------------------------------------------------------------------
def render_diagram(visual_type: str, title: str, code: str):
    st.markdown(f"##### 🖼️ {title}")
    with st.container(border=True):
        try:
            if visual_type == "graphviz" or "digraph" in code or "graph " in code:
                clean_dot = code.strip().replace("```dot", "").replace("```", "").strip()
                st.graphviz_chart(clean_dot, use_container_width=True)
            elif visual_type == "mermaid":
                clean_mermaid = code.strip().replace("```mermaid", "").replace("```", "").strip()
                st.markdown(f"```mermaid\n{clean_mermaid}\n```")
            else:
                st.markdown(code)
        except Exception as err:
            st.warning(f"Diagram notice: {str(err)}")
            st.code(code)


# -----------------------------------------------------------------------------
# Document Initialization & Ingestion Helpers
# -----------------------------------------------------------------------------
def ingest_pdf_bytes(pdf_bytes: bytes, file_name: str) -> Tuple[int, int, str]:
    """Ingests, chunks, embeds and indexes PDF bytes into vector storage."""
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]
    chunks, pages, method = extract_and_chunk_pdf_safe(pdf_bytes)
    embed_model = get_cached_embedding_model()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(chunk_texts, batch_size=32, show_progress_bar=False).tolist()

    is_pc = False
    if PINECONE_API_KEY and PINECONE_AVAILABLE:
        try:
            pc_store = SafePineconeStore(api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME)
            pc_store.upsert(file_hash, chunks, embeddings)
            is_pc = True
        except Exception:
            upsert_local_vector_db(file_hash, chunks, embeddings)
    else:
        upsert_local_vector_db(file_hash, chunks, embeddings)

    st.session_state.active_doc_id = file_hash
    st.session_state.active_doc_name = file_name
    st.session_state.total_pages = pages
    st.session_state.total_chunks = len(chunks)
    st.session_state.is_pinecone = is_pc
    return len(chunks), pages, method


def initialize_benchmark_paper():
    """Initializes the built-in 'Attention Is All You Need' benchmark paper into vector DB."""
    embed_model = get_cached_embedding_model()
    texts = [c["text"] for c in BENCHMARK_PAPER_CHUNKS]
    embeddings = embed_model.encode(texts).tolist()

    upsert_local_vector_db("demo_transformer", BENCHMARK_PAPER_CHUNKS, embeddings)
    is_pc = False
    if PINECONE_API_KEY and PINECONE_AVAILABLE:
        try:
            pc_store = SafePineconeStore(api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME)
            pc_store.upsert("demo_transformer", BENCHMARK_PAPER_CHUNKS, embeddings)
            is_pc = True
        except Exception:
            pass

    st.session_state.active_doc_id = "demo_transformer"
    st.session_state.active_doc_name = "Attention Is All You Need (Vaswani et al.)"
    st.session_state.total_pages = 6
    st.session_state.total_chunks = len(BENCHMARK_PAPER_CHUNKS)
    st.session_state.is_pinecone = is_pc


# Auto-load benchmark paper on first run so the studio is immediately active
if "active_doc_id" not in st.session_state:
    initialize_benchmark_paper()


# -----------------------------------------------------------------------------
# Sidebar: Document Vault & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 ScholarRAG Studio")
    st.caption("Enterprise Academic Vector Search & Synthesis")

    # System Status Indicator
    with st.container(border=True):
        st.markdown("**🛡️ System Status**")
        if OPENROUTER_API_KEY:
            st.markdown('<span style="color:#818cf8; font-weight:600; font-size:0.82rem;">● OpenRouter LLM Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#fbbf24; font-weight:600; font-size:0.82rem;">⚠ Missing OPENROUTER_API_KEY</span>', unsafe_allow_html=True)

        if PINECONE_API_KEY and PINECONE_AVAILABLE:
            st.markdown('<span style="color:#34d399; font-weight:600; font-size:0.82rem;">● Pinecone Serverless Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#38bdf8; font-weight:600; font-size:0.82rem;">● Local Vector Engine Active</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📄 Research Document Vault**")

    # One-Click Fast Load Buttons
    st.caption("Pre-packaged Papers (One-Click):")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("📚 Transformer", use_container_width=True):
            initialize_benchmark_paper()
            st.rerun()
    with col_d2:
        if st.button("🚀 Elon Musk Case", use_container_width=True):
            elon_path = Path(__file__).parent / "elon_musk_case_study.pdf"
            if elon_path.exists():
                with open(elon_path, "rb") as f:
                    ingest_pdf_bytes(f.read(), "Elon Musk Engineering Case Study")
                st.rerun()

    uploaded_pdf = st.file_uploader("Or Upload Custom Research PDF", type=["pdf"])

    # Document Ingestion Handler
    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.read()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]

        if st.session_state.get("active_doc_id") != file_hash:
            with st.status(f"Ingesting '{uploaded_pdf.name}'...", expanded=True) as status:
                try:
                    chunks_count, pages_count, method = ingest_pdf_bytes(pdf_bytes, uploaded_pdf.name)
                    method_label = "Neural OCR" if method == "neural-ocr" else ("Academic Markdown" if method == "academic-markdown" else "Direct Streams")
                    status.update(label=f"✅ Ready! ({chunks_count} chunks via {method_label})", state="complete")
                    st.rerun()
                except Exception as ingest_err:
                    status.update(label=f"❌ Ingestion notice: {str(ingest_err)}", state="error")
                    st.error(str(ingest_err))

    st.markdown("---")
    st.markdown("**⚙️ Synthesis Settings**")
    model_choice = st.selectbox(
        "LLM Reasoner",
        options=[
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
            "mistralai/mistral-nemo",
            "qwen/qwen-2.5-72b-instruct"
        ],
        index=0
    )
    top_k = st.slider("Retrieved Evidence Chunks (Top-K)", min_value=2, max_value=8, value=4)

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main Application Header & Live Metrics
# -----------------------------------------------------------------------------
st.markdown("""
<div class="brand-container">
    <div>
        <div class="brand-title">⚡ ScholarRAG Studio</div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 3px;">
            Production Academic Vector Retrieval, Grounded Synthesis & Neural Architecture Diagrams
        </div>
    </div>
    <div>
        <span class="brand-badge">Enterprise v2.4</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Active Paper Intelligence Deck
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Active Research Paper</div>
        <div class="stat-value" style="font-size: 1.05rem;" title="{st.session_state.get('active_doc_name', 'None')}">
            {st.session_state.get('active_doc_name', 'None')[:22]}...
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Document Pages</div>
        <div class="stat-value">{st.session_state.get('total_pages', 0)} Pages</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Indexed Vector Chunks</div>
        <div class="stat-value">{st.session_state.get('total_chunks', 0)} Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    v_type = "Pinecone Serverless" if st.session_state.get("is_pinecone") else "In-Memory Vector DB"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Vector Store Engine</div>
        <div class="stat-value" style="font-size: 1.05rem;">{v_type}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PROMINENT QUESTION CONSOLE (Front & Center!)
# -----------------------------------------------------------------------------
st.markdown('<div class="question-panel">', unsafe_allow_html=True)
st.markdown("##### 💬 Research Inquiry & Architecture Generation")
st.caption("Ask questions about equations, model architecture, baseline benchmarks, or ablation studies.")

with st.form(key="research_inquiry_form", clear_on_submit=False):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "Research Question",
            placeholder="Type any question (e.g. Explain Multi-Head Attention and draw the neural diagram)...",
            label_visibility="collapsed"
        )
    with col_btn:
        form_submitted = st.form_submit_button("🚀 Analyze", use_container_width=True, type="primary")

# Instant Research Inquiries (One-Click Chips)
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
st.caption("⚡ Instant Analysis Chips (Click to run):")
chip_cols = st.columns(4)
chip_query = None

is_elon = "elon" in st.session_state.get("active_doc_name", "").lower()
if is_elon:
    if chip_cols[0].button("🚀 SpaceX Falcon & Starship", use_container_width=True):
        chip_query = "Explain SpaceX's reusable rocketry architecture (Falcon 9 & Starship) and generate an orbital pipeline flowchart."
    if chip_cols[1].button("⚡ Tesla Vision FSD", use_container_width=True):
        chip_query = "How does Tesla's vision-only Full Self-Driving neural pipeline work? Generate an architecture diagram."
    if chip_cols[2].button("🧠 Neuralink N1 Interface", use_container_width=True):
        chip_query = "Summarize the Neuralink N1 brain-computer interface and its ultra-fine thread architecture."
    if chip_cols[3].button("📐 5-Step Algorithm", use_container_width=True):
        chip_query = "Explain Elon Musk's 5-step engineering algorithm and First Principles Thinking with a flowchart."
else:
    if chip_cols[0].button("📐 Neural Architecture", use_container_width=True):
        chip_query = "Explain the complete Transformer neural architecture and generate a detailed architecture diagram."
    if chip_cols[1].button("🔬 Multi-Head Attention", use_container_width=True):
        chip_query = "How does Scaled Dot-Product and Multi-Head Attention work mathematically? Generate a flowchart."
    if chip_cols[2].button("📊 Benchmark Results", use_container_width=True):
        chip_query = "What are the primary translation benchmark scores (BLEU) on WMT 2014 and training efficiencies?"
    if chip_cols[3].button("⚙️ Positional Encoding", use_container_width=True):
        chip_query = "Why is sinusoidal positional encoding used and what are its mathematical formulas?"

st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Conversation & Execution Pipeline
# -----------------------------------------------------------------------------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

active_query = chip_query or (user_query.strip() if form_submitted and user_query.strip() else None)

if active_query:
    if not OPENROUTER_API_KEY:
        st.error("⚠️ OPENROUTER_API_KEY is not set in your .env file.")
    else:
        with st.spinner("Retrieving vector chunks & generating architecture diagram..."):
            # 1. Embed query locally (sub-50ms)
            embed_model = get_cached_embedding_model()
            q_emb = embed_model.encode([active_query])[0].tolist()

            # 2. Retrieve relevant chunks
            doc_id = st.session_state.get("active_doc_id", "demo_transformer")
            if st.session_state.get("is_pinecone") and PINECONE_API_KEY:
                pc_store = SafePineconeStore(api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME)
                retrieved_chunks = pc_store.search(doc_id, q_emb, top_k=top_k)
            else:
                retrieved_chunks = query_local_vector_db(doc_id, q_emb, top_k=top_k)

            # 3. LLM Synthesis with Diagram Generation
            llm_client = get_llm_client(OPENROUTER_API_KEY)
            result_data = synthesize_with_visuals(active_query, retrieved_chunks, llm_client, model_choice)
            result_data["retrieved_chunks"] = retrieved_chunks

            # Store in session state
            st.session_state.conversation_history.insert(0, {
                "query": active_query,
                "data": result_data
            })


# -----------------------------------------------------------------------------
# Results Deck & Visual Studio
# -----------------------------------------------------------------------------
if st.session_state.conversation_history:
    st.markdown("---")
    latest = st.session_state.conversation_history[0]
    res = latest["data"]

    st.markdown(f"#### 🔬 Analysis: *{latest['query']}*")
    tabs = st.tabs(["🖼️ Visual Architecture Diagram", "📝 Academic Synthesis", "📚 Grounded Source Evidence"])

    with tabs[0]:
        render_diagram(
            visual_type=res.get("visual_type", "graphviz"),
            title=res.get("visual_title", "Neural Architecture Diagram"),
            code=res.get("visual_code", "")
        )

    with tabs[1]:
        st.markdown(res.get("answer", "No synthesis provided."))

    with tabs[2]:
        chunks = res.get("retrieved_chunks", [])
        st.caption(f"Retrieved {len(chunks)} high-confidence contextual chunks from paper:")
        for c_idx, c in enumerate(chunks):
            score_val = c.get("score", 0.0)
            st.markdown(f"""
            <div class="source-card">
                <span class="score-pill">Cosine Similarity: {score_val:.3f}</span>
                <strong>Source Chunk #{c_idx + 1} — Page {c['page']}</strong><br/>
                <div style="margin-top: 6px;">{c['text']}</div>
            </div>
            """, unsafe_allow_html=True)

    if len(st.session_state.conversation_history) > 1:
        st.markdown("##### 📜 Previous Inquiries in this Session")
        for prev in st.session_state.conversation_history[1:]:
            with st.expander(f"Inquiry: {prev['query']}"):
                p_res = prev["data"]
                st.markdown(p_res.get("answer", ""))
                render_diagram(p_res.get("visual_type", "graphviz"), p_res.get("visual_title", "Diagram"), p_res.get("visual_code", ""))
else:
    # First-run state: show a welcome card with pre-rendered diagram demonstration
    st.markdown("#### 🖼️ Visual Architecture Studio Demonstration")
    with st.container(border=True):
        st.markdown("**Transformer Encoder-Decoder Neural Flowchart** *(Auto-Generated Preview)*")
        demo_dot = """
        digraph TransformerArchitecture {
            rankdir=TB;
            node [shape=box, style="filled,rounded", fillcolor="#1e293b", fontcolor="#f8fafc", fontname="Helvetica", color="#6366f1", penwidth=1.5];
            edge [color="#94a3b8", fontcolor="#cbd5e1", fontname="Helvetica"];

            Inputs [label="Inputs\n(Tokens)", fillcolor="#312e81"];
            InputEmb [label="Input Embedding +\nPositional Encoding"];
            EncoderBlock [label="Encoder Stack (N=6)\n• Multi-Head Self-Attention\n• Add & Norm\n• Feed Forward\n• Add & Norm"];
            
            Outputs [label="Outputs\n(Shifted Right)", fillcolor="#312e81"];
            OutputEmb [label="Output Embedding +\nPositional Encoding"];
            DecoderBlock [label="Decoder Stack (N=6)\n• Masked Multi-Head Attention\n• Multi-Head Cross-Attention\n• Feed Forward & Norm"];
            
            LinearSoftmax [label="Linear +\nSoftmax", fillcolor="#312e81"];
            OutputProbs [label="Output Probabilities", fillcolor="#047857"];

            Inputs -> InputEmb -> EncoderBlock;
            Outputs -> OutputEmb -> DecoderBlock;
            EncoderBlock -> DecoderBlock [label="Keys, Values", style=dashed, color="#a78bfa"];
            DecoderBlock -> LinearSoftmax -> OutputProbs;
        }
        """
        st.graphviz_chart(demo_dot, use_container_width=True)
        st.caption("💡 Use the inquiry box above or click any research chip to generate grounded diagrams for your paper.")
