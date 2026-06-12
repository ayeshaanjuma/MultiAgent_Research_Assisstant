# Multi-Agent Research Assistant

A professional, high-performance research assistant built with Groq, Tavily, LangChain, ChromaDB, HuggingFace embeddings, Streamlit, ReportLab, and Jinja2.

## Features

- **Multi-Agent Orchestration**: Transparent agent dashboard with process workspaces for the Research Agent, Summarization Agent, and Fact Verification Agent.
- **Redesigned Report Generator UI**:
  - Eliminated raw HTML outputs and custom style leakages for a clean, distraction-free environment.
  - **Compact Report Status Card**: Dynamically displays compilation status (`Complete ✅`, `Generating ⏳`, `Failed ❌`), sources used, verified claims, and confidence scores directly below the agent workflow.
  - **Embedded Professional Document Preview**: Rendered directly on the main page in a comfortable reading layout using native Streamlit containers and clean expanders.
  - **Smooth Navigation**: Anchor scroll link with hardware-accelerated CSS transitions to seamlessly guide users from the workflow directly to the report preview.
- **Parallel Search & Local RAG**: Tavily-powered web search, PubMed integration, and vector search on uploaded PDFs via local ChromaDB.
- **Async Summarisation**: Batch Groq LLM processing consolidated into comprehensive executive summaries.
- **Smart Fact Verification**: Extracts factual statements and performs credibility scoring with external cross-referencing.
- **Export Formats**: Standardized download section supporting side-by-side exports for PDF, DOCX, Markdown, and TXT.

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set your keys:

```powershell
copy .env.example .env
```

4. Run the app:

```powershell
streamlit run streamlit_app.py
```

## Architecture Notes

- **Optimized Client Routing**: The Groq API client (`utils/clients.py`) is optimized to query the OpenAI-compatible endpoint first for lower latency and better error tracking.
- **ChromaDB**: Utilizes a persistent local database under `db/` or `chroma_db/`.
- **Theme & Design**: Modern dark-theme glassmorphism customized globally via CSS overrides for all bordered containers and expanders.
