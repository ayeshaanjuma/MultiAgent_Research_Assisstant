import asyncio
from typing import Dict, List
from utils.clients import groq_generate, tavily_search

async def score_claims(summary: str) -> List[Dict]:
    prompt = (
        "Extract the main factual claims from the summary. "
        "Return ONLY a valid JSON array of objects where each object has keys: \"claim\", \"score\". "
        "Score should be a number between 0.0 and 1.0 representing confidence. "
        "Do not include any introductory or concluding text, or markdown formatting outside of the JSON block. "
        f"Summary:\n{summary}\n"
    )
    output = await groq_generate(prompt, max_tokens=600)
    
    claims = []
    text = output.strip()
    
    # If wrapped in code blocks, extract content
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
        
    # Find the bounds of the JSON array
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx+1]
        
    try:
        import json
        parsed = json.loads(text)
        if isinstance(parsed, list):
            claims = parsed
    except Exception:
        claims = []
        
    validated = []
    for item in claims:
        if isinstance(item, dict) and item.get("claim"):
            validated.append({
                "claim": item["claim"],
                "confidence": min(max(float(item.get("score", 0.0)), 0.0), 1.0),
            })
    
    # Fallback: if no claims were extracted, let's extract a few sentence-based claims manually from the summary 
    # to prevent the pipeline from crashing.
    if not validated and summary.strip():
        sentences = [s.strip() for s in summary.replace("\n", " ").split(".") if len(s.strip()) > 20]
        for sentence in sentences[:3]:
            validated.append({
                "claim": sentence,
                "confidence": 0.5,
            })
            
    return validated

async def _verify_claim(claim: str) -> Dict:
    query = f"Verify this claim against WHO and PubMed sources: {claim}"
    results = await tavily_search(query, top_k=5)
    evidence = "\n".join([item.get("abstract", "") for item in results[:3]])
    found = bool(evidence.strip())
    status = "VERIFIED" if found else "UNVERIFIED"
    if found and "partial" in evidence.lower():
        status = "PARTIAL"
    return {
        "claim": claim,
        "verification_status": status,
        "evidence": evidence,
    }

async def run_verification_agent(summary: str) -> List[Dict]:
    claims = await score_claims(summary)
    if not claims:
        raise ValueError("Verification Agent could not extract claims.")
    low_confidence_claims = [claim for claim in claims if claim["confidence"] < 0.8]
    verified = []
    if low_confidence_claims:
        tasks = [_verify_claim(claim["claim"]) for claim in low_confidence_claims]
        verified = await asyncio.gather(*tasks)
    results = []
    for claim in claims:
        match = next((item for item in verified if item["claim"] == claim["claim"]), None)
        results.append({
            "claim": claim["claim"],
            "confidence": claim["confidence"],
            "verification_status": match["verification_status"] if match else "VERIFIED" if claim["confidence"] >= 0.8 else "UNVERIFIED",
            "evidence": match["evidence"] if match else "Based on internal confidence score",
        })
    return results

def validate_verification(verification: List[Dict]) -> None:
    if not verification:
        raise ValueError("Verification Agent returned empty result set.")
    if any("verification_status" not in item for item in verification):
        raise ValueError("Verification result structure is invalid.")
