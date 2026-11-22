# 🎓 Academic RAG Studio (Production-Ready)

A production-grade Academic Research RAG system built with **Streamlit**, **Pinecone Vector Database**, **SentenceTransformers**, and **OpenRouter LLMs**, featuring automated **visual diagram generation** (Graphviz architectures, Mermaid workflows, and benchmark tables).

---

## 🌟 Key Features

- **Pinecone Serverless Integration**: Isolated per-document namespaces with sub-50ms cosine similarity vector retrieval.
- **Automated Visual Suggestions**: LLM determines when an academic answer warrants a visual representation and generates:
  - **Graphviz (`DOT`)** for model architectures and neural network block diagrams.
  - **Mermaid** for sequence diagrams, state machines, and algorithmic decision flows.
  - **Markdown Tables** for benchmark comparisons and ablation studies.
- **Multi-Tab Academic Interface**:
  - 📝 **Synthesis**: Rigorous academic explanation with page-level citations (`[Page X]`).
  - 📊 **Visual Representation**: Interactive, client-side rendered architectural diagrams.
  - 📚 **Grounded Sources**: Expandable cards displaying exact retrieved context, similarity scores, and page numbers.
- **Decoupled Ingestion Pipeline**: Ingests and embeds the PDF once with SHA-256 deduplication and sliding window chunking with overlap.
- **Fail-Safe Resilience**: If Pinecone is not yet configured, the system seamlessly falls back to a high-speed namespaced in-memory vector store.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (.env)
Copy the example environment file and add your OpenRouter (and optional Pinecone) API key:
```bash
cp .env.example .env
```
In `.env`:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here   # Optional (falls back to local vector memory if omitted)
PINECONE_INDEX_NAME=academic-rag
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 🛠️ Architecture Overview

```
User Query + Uploaded Academic PDF
                │
                ▼
     [ SHA-256 Check & Ingestion ]
                │
   ┌────────────┴────────────┐
   ▼                         ▼
[ Pinecone Serverless ]  [ Local Vector DB (Fallback) ]
   └────────────┬────────────┘
                │
         [ Similarity Search ]
                │
                ▼
  [ OpenRouter LLM (Structured JSON) ]
  ├── 1. Academic Answer + Page Citations
  ├── 2. Visual Type (Graphviz / Mermaid / Table)
  └── 3. Clean Diagram Definition Code
                │
                ▼
[ Streamlit Interactive Tabbed UI ]
```
