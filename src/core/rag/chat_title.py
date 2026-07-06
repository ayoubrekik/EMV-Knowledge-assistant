import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = "qwen3:8b" #os.getenv("MODEL_NAME", "qwen3:8b")


def generate_title(question: str) -> str:

    prompt = f"""
Generate a short conversation title from this user input.

Strict Rules:
- Maximum 4 words
- Best : 3 Words
- No quotes
- No punctuation at the end
- Use only the main topic
- Do not answer the question
- No special Characters
Question:
{question}
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        title = data.get("response", "").strip()

        if not title:
            return question[:32]

        return title

    except Exception as e:
        print("Title generation error:", e)
        return question[:32]