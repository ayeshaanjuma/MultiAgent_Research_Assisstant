import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def test():
    urls = [
        "https://api.groq.com/v1/chat/completions",
        "https://api.groq.com/openai/v1/chat/completions",
        "https://api.groq.com/v1/llm/text-generation"
    ]
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }
    
    async with httpx.AsyncClient() as client:
        for url in urls:
            print(f"Testing {url}...")
            resp = await client.post(url, headers=headers, json=payload)
            print("Status code:", resp.status_code)
            print("Response:", resp.text)
            print("-" * 40)

asyncio.run(test())
