from typing import List

from langchain_core.prompts import ChatPromptTemplate

from .history import format_history_for_prompt
from .prompts import ROUTER_PROMPT, CONTEXTUAL_REWRITE_PROMPT
from .types import InputType


def normalize_label(label: str) -> InputType:
    raw = (label or "").strip().lower()
    allowed = list(InputType.__args__)  # type: ignore[attr-defined]

    if raw in allowed:
        return raw  # type: ignore[return-value]

    for item in allowed:
        if item in raw:
            return item  # type: ignore[return-value]

    return "noise"


def classify_input(llm, question: str, history: List) -> InputType:
    if not question or not question.strip():
        return "noise"

    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)

    messages = prompt.invoke({
        "chat_history": format_history_for_prompt(history, max_messages=6),
        "question": question,
    })

    result = llm.invoke(messages)
    return normalize_label(result.content)


def rewrite_for_retrieval(
    llm,
    question: str,
    history: list,
    current_topic: str | None = None,
) -> str:
    prompt = ChatPromptTemplate.from_template(CONTEXTUAL_REWRITE_PROMPT)

    chat_history = format_history_for_prompt(history, max_messages=4)

    print("CURRENT TOPIC:", current_topic)
    print("CHAT HISTORY SENT TO REWRITER:")
    print(chat_history)
    print("QUESTION SENT TO REWRITER:", question)

    messages = prompt.invoke({
        "current_topic": current_topic or "No active topic.",
        "chat_history": chat_history,
        "question": question,
    })

    result = llm.invoke(messages)
    rewritten = (result.content or "").strip()

    print("RAW REWRITE OUTPUT:", rewritten)

    return rewritten if rewritten else question
    
def is_definition_question(question: str) -> bool:
    q = question.strip().lower()

    return (
        q.startswith("what is ")
        or q.startswith("what are ")
        or q.startswith("define ")
        or q.startswith("meaning of ")
        or q.startswith("explain ")
    )


def has_direct_definition_match(doc, question: str) -> bool:
    text = (doc.page_content or "").lower()
    q = question.lower()

    target = (
        q.replace("what is", "")
        .replace("what are", "")
        .replace("define", "")
        .replace("meaning of", "")
        .replace("explain", "")
        .replace("?", "")
        .strip()
    )

    return bool(target and target in text[:1000])