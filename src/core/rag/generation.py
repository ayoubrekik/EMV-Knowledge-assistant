from langchain_core.prompts import ChatPromptTemplate
from .types import InputType

from src.core.rag.history import format_history_for_prompt

from src.core.rag.prompts import (
    EMV_RAG_PROMPT,
    HEX_RAG_PROMPT,
    SOURCE_LOOKUP_PROMPT,
    FOLLOWUP_PROMPT,
    TAG_LOOKUP_PROMPT,
    DEFINITION_PROMPT,
    COMPARISON_PROMPT,
)


def build_non_rag_answer(llm, question: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
You are an EMV specification assistant.

Rules:
- If the message is a greeting, thanks, or polite conversation, respond briefly and naturally.
- If the message is unrelated to EMV specifications, payment systems, smart cards, APDU commands, EMV tags, or the EMV books, politely refuse.
- Do not answer questions that require external knowledge or are unrelated to EMV specifications.
- Do NOT answer unrelated factual questions.
- Keep responses short.

User message:
{question}

Answer:
""")
    messages = prompt.invoke({"question": question})
    response = llm.invoke(messages)
    return response.content or ""


def stream_answer(llm, context: str, question: str, input_type: str, history: list):
    if input_type == "hex_decode_question":
        template = HEX_RAG_PROMPT
    elif input_type == "source_lookup":
        template = SOURCE_LOOKUP_PROMPT
    elif input_type == "contextual_follow_up":
        template = FOLLOWUP_PROMPT
    elif input_type == "tag_lookup_question":
        template = TAG_LOOKUP_PROMPT
    elif input_type == "definition_question":
        template = DEFINITION_PROMPT
    elif input_type == "comparison_question":
        template = COMPARISON_PROMPT
    else:
        template = EMV_RAG_PROMPT

    prompt = ChatPromptTemplate.from_template(template)

    messages = prompt.invoke({
        "context": context,
        "question": "/no_think\n" + question,
        "chat_history": format_history_for_prompt(history, max_messages=6),
    })

    return llm.stream(messages)