# Multi-Agent Research Assistant

A free-stack research assistant built with Groq, Tavily, LangChain, ChromaDB, HuggingFace embeddings, Streamlit, ReportLab, and Jinja2.

## Features

- Parallel web search with Tavily, semantic search via ChromaDB, and uploaded PDF search
- Async summarisation with LangChain `load_summarize_chain(chain_type="map_reduce")`
- Smart fact verification that only hits external sources for low-confidence claims
- Jinja2-powered report generator with PDF export
- Streamlit UI with query input, PDF drag-and-drop, progress tracking and download

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

## Notes

- The project uses a local ChromaDB store under `db/`.
- Uploaded PDFs are indexed in memory and queried as part of Agent 1.
- Replace API endpoints in `utils/clients.py` if Tavily or Groq update their service URLs.
- Use Streamlit Cloud for free deployment.
