import asyncio
from typing import List
from utils.clients import groq_generate

async def _summarize_batch(batch: List[dict]) -> str:
    prompt = (
        "You are a research summarisation assistant. Read the following article abstracts and create a concise, coherent summary for each entry. "
        "Return only the summaries, separated by new lines."
        "\n\nAbstracts:\n"
        + "\n\n".join([f"- {item['title']}: {item['abstract']}" for item in batch])
    )
    output = await groq_generate(prompt, max_tokens=600)
    return output.strip()

async def run_summarisation_agent(articles: List[dict]) -> str:
    if not articles:
        raise ValueError("Summarisation Agent received no articles.")

    batch_size = 5
    batch_summaries = []
    for index in range(0, len(articles), batch_size):
        batch = articles[index : index + batch_size]
        batch_summary = await _summarize_batch(batch)
        batch_summaries.append(batch_summary)

    merge_prompt = (
        "You are a research assistant. Merge the following partial summaries into one coherent, structured summary of the research findings, "
        "aiming for an output that would cover approximately 5 pages when expanded into a report. Use paragraphs, transitions, and keep it professional."
        "\n\nPartial summaries:\n"
        + "\n\n".join(batch_summaries)
    )
    final_summary = await groq_generate(merge_prompt, max_tokens=1200)
    return final_summary.strip()

def validate_summary(summary: str) -> None:
    if not summary or len(summary) < 200:
        raise ValueError("Summary is too short or empty.")
