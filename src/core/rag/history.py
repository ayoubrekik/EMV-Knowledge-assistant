from typing import List
from langchain_core.messages import AIMessage, HumanMessage

chat_histories = {}

def get_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]

def clear_conversation(session_id: str = "default"):
    chat_histories[session_id] = []

def format_history_for_prompt(history: List, max_messages: int = 6) -> str:
    if not history:
        return "No previous conversation."

    recent = history[-max_messages:]
    lines = []

    for message in recent:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"Assistant: {message.content}")

    return "\n".join(lines)
