import asyncio
from typing import Dict, List, Optional
from utils.clients import tavily_search, load_pdf_text
from rag_pipeline import RAGPipeline

async def _search_tavily(query: str, top_k: int = 10) -> List[Dict]:
    return await tavily_search(query, top_k=top_k)

async def _search_semantic(rag: RAGPipeline, query: str) -> List[Dict]:
    return rag.hybrid_search(query, k=20)

async def _search_uploaded_pdf(rag: RAGPipeline, query: str) -> List[Dict]:
    return rag.pdf_search(query, k=20)

def _normalize_results(results: List[Dict], source_label: str) -> List[Dict]:
    normalized = []
    for item in results:
        normalized.append({
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
            "source": source_label,
            "date": item.get("date", ""),
            "abstract": item.get("abstract", "")[:320],
            "score": item.get("score", 0.0),
        })
    return normalized

async def run_research_agent(
    query: str,
    rag_pipeline: RAGPipeline,
    uploaded_pdf_text: Optional[str] = None,
) -> List[Dict]:
    tasks = [
        asyncio.create_task(_search_tavily(query, top_k=20)),
        asyncio.create_task(_search_semantic(rag_pipeline, query)),
    ]
    if uploaded_pdf_text:
        tasks.append(asyncio.create_task(_search_uploaded_pdf(rag_pipeline, query)))
    else:
        tasks.append(asyncio.create_task(asyncio.sleep(0, result=[])))

    tavily_results, semantic_results, pdf_results = await asyncio.gather(*tasks)

    combined = []
    combined.extend(_normalize_results(tavily_results, "Tavily Web"))
    combined.extend(_normalize_results(semantic_results, "Chroma Semantic"))
    combined.extend(_normalize_results(pdf_results, "Uploaded PDF"))

    unique = {}
    for item in combined:
        key = (item["url"], item["title"], item["abstract"][:120])
        if key not in unique or item["score"] > unique[key]["score"]:
            unique[key] = item

    ranked = sorted(unique.values(), key=lambda x: x.get("score", 0), reverse=True)
    return ranked[:20]

def validate_research_results(results: List[Dict]) -> None:
    if not results:
        raise ValueError("Research Agent returned no results.")
    
    # At least some results should have non-empty abstracts
    results_with_abstracts = sum(1 for item in results if item.get("abstract", "").strip())
    if results_with_abstracts == 0:
        raise ValueError("No research articles have abstracts. Check API configuration.")
