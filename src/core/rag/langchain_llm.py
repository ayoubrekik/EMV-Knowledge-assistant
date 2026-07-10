import os
from langchain_ollama import ChatOllama


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = "qwen3:8b" #os.getenv("OLLAMA_MODEL", "qwen3:8b")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def get_llm(temp=0.2):
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temp,
        num_ctx=4096,
        reasoning=False,
    )

