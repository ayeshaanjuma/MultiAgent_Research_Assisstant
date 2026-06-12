import json
import httpx
import chromadb
from typing import Any, Dict, List
from PyPDF2 import PdfReader
from config import GROQ_API_KEY, TAVILY_API_KEY, CHROMADB_MODE, CHROMADB_PATH, CHROMADB_HOST, CHROMADB_PORT

GROQ_API_URLS = [
    "https://api.groq.com/openai/v1/chat/completions",
    "https://api.groq.com/v1/chat/completions",
    "https://api.groq.com/v1/llm/text-generation",
]
TAVILY_API_URL = "https://api.tavily.com/search"


async def groq_generate(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """
    Robust Groq text generation helper.

    Tries multiple Groq endpoints and payload formats to handle version differences.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    primary_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=60) as client:
        for idx, url in enumerate(GROQ_API_URLS):
            try:
                if url.endswith("/llm/text-generation"):
                    payload = {
                        "model": "llama-3.3-70b-versatile",
                        "input": prompt,
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    }
                else:
                    # OpenAI-compatible chat endpoints
                    payload = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }

                resp = await client.post(url, headers=headers, json=payload)
                # If 404/400, try next format/endpoint
                if resp.status_code >= 400:
                    exc = httpx.HTTPStatusError(f"bad response: {resp.status_code} {resp.text}", request=resp.request, response=resp)
                    if idx == 0:
                        primary_exc = exc
                    continue

                data = resp.json()
                # Parse multiple possible response shapes
                if isinstance(data, dict):
                    # OpenAI-like
                    if "choices" in data and isinstance(data["choices"], list):
                        first = data["choices"][0]
                        if isinstance(first, dict):
                            if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                                return first["message"]["content"]
                            if "text" in first:
                                return first["text"]
                    # Groq older formats
                    if "output" in data:
                        return data["output"]
                    if "result" in data:
                        return data["result"]
                    if "outputs" in data and isinstance(data["outputs"], list):
                        first = data["outputs"][0]
                        if isinstance(first, dict) and "content" in first:
                            return first["content"]
                        return first

                return json.dumps(data)
            except Exception as exc:
                if idx == 0:
                    primary_exc = exc
                continue

    # If we reach here, all attempts failed
    if primary_exc:
        raise primary_exc
    raise RuntimeError("Unexpected failure calling Groq API")

def get_chromadb_client() -> chromadb.ClientAPI:
    """
    Factory function to initialize ChromaDB client based on configuration.
    
    Supports three modes:
    - ephemeral: In-memory database (no persistence)
    - persistent: Local file-based storage (default)
    - http: Remote Chroma server
    
    Returns:
        chromadb.ClientAPI: Configured ChromaDB client
    """
    if CHROMADB_MODE == "ephemeral":
        return chromadb.EphemeralClient()
    elif CHROMADB_MODE == "http":
        return chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    else:  # persistent (default)
        return chromadb.PersistentClient(path=CHROMADB_PATH)


async def tavily_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "top_k": top_k,
        "include_answer": True,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(TAVILY_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("results", []):
        # Try multiple fields for abstract content
        abstract = item.get("snippet") or item.get("answer") or item.get("content") or ""
        # If still no abstract, use title as fallback
        if not abstract:
            abstract = item.get("title", "")
        results.append({
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
            "source": "Tavily Web",
            "date": item.get("published_date", ""),
            "abstract": abstract[:320] if abstract else "",
            "score": float(item.get("score", 0)),
        })
    return results

def load_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
