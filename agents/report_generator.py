import json
import os
from typing import Dict, List
from jinja2 import Environment, BaseLoader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from utils.clients import groq_generate

REPORT_TEMPLATE = """
Abstract
{{ abstract }}

Introduction
{{ introduction }}

Literature Review
{{ literature_review }}

Findings
{{ findings }}

Conclusion
{{ conclusion }}

References
{{ references }}
"""

async def run_report_generator(
    query: str,
    summary: str,
    verification: List[Dict],
    articles: List[Dict],
) -> Dict[str, str]:
    prompt = (
        "Generate a structured research report with six sections: Abstract, Introduction, "
        "Literature Review, Findings, Conclusion, References. Use the query, summary, verified claims, "
        "and article metadata. Return valid JSON with keys abstract, introduction, literature_review, "
        "findings, conclusion, references."
        f"\n\nQuery:\n{query}\n\nSummary:\n{summary}\n\nVerification:\n{json.dumps(verification, indent=2)}\n\nArticles:\n{json.dumps(articles[:10], indent=2)}"
    )
    output = await groq_generate(prompt, max_tokens=1200)
    try:
        report_data = json.loads(output)
    except Exception:
        # Fallback: wrap output in sections using text parsing heuristic
        report_data = {
            "abstract": summary[:500],
            "introduction": summary[:800],
            "literature_review": "\n".join([a["abstract"] for a in articles[:5]]),
            "findings": summary,
            "conclusion": summary[-500:],
            "references": "\n".join([f"- {item.get('title')} ({item.get('url')})" for item in articles[:10]]),
        }
    return report_data

def render_report(report_data: Dict[str, str]) -> str:
    env = Environment(loader=BaseLoader())
    template = env.from_string(REPORT_TEMPLATE)
    return template.render(**report_data)

def export_report_pdf(report_text: str, filename: str) -> str:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    body = []
    for block in report_text.split("\n\n"):
        body.append(Paragraph(block.replace("\n", "<br/>"), styles["BodyText"]))
        body.append(Spacer(1, 12))
    doc.build(body)
    return filename

def validate_report(report_data: Dict[str, str]) -> None:
    required = ["abstract", "introduction", "literature_review", "findings", "conclusion", "references"]
    if any(key not in report_data or not report_data[key].strip() for key in required):
        raise ValueError("Report Generator produced malformed report content.")
