import time
from typing import Dict, List, Literal

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.core.rag.langchain_chroma import get_langchain_chroma
from src.core.rag.langchain_llm import get_llm


chat_histories: Dict[str, List] = {}


InputType = Literal[
    "standalone_emv_question",
    "contextual_follow_up",
    "definition_question",
    "comparison_question",
    "procedure_question",
    "source_lookup",
    "citation_request",
    "summarization_request",
    "clarification_request",
    "greeting",
    "gratitude",
    "small_talk",
    "out_of_scope_question",
    "project_question",
    "frontend_command",
    "empty_input",
    "noise",
]


RAG_INPUT_TYPES = {
    "standalone_emv_question",
    "definition_question",
    "comparison_question",
    "procedure_question",
    "source_lookup",
    "citation_request",
    "summarization_request",
    "clarification_request",
}


ROUTER_PROMPT = """
You are an input router for an EMV technical RAG chatbot.

Classify the latest user input into exactly ONE category.

Categories:

- standalone_emv_question:
  A clear EMV-related question that can be answered directly.
  Example: "Explain offline data authentication."

- contextual_follow_up:
  A follow-up that depends on chat history.
  Example: "explain more", "what about this?", "give more details", "why?", "continue".

- definition_question:
  The user asks what an EMV/payment term means.
  Example: "What is SDA?", "Define DDA", "What does AFL mean?"

- comparison_question:
  The user asks to compare EMV/payment concepts.
  Example: "What is the difference between SDA and DDA?"

- procedure_question:
  The user asks for steps, flow, structure, sequence, or process.
  Example: "Describe the SELECT command APDU structure."

- source_lookup:
  The user asks where something is found in the documents.
  Example: "Where is SDA described?", "Which section talks about CAPK?"

- citation_request:
  The user asks for sources, references, pages, sections, or citations.
  Example: "Give me the source", "Show citations", "Which page?"

- summarization_request:
  The user asks to summarize EMV content.
  Example: "Summarize terminal risk management."

- clarification_request:
  The user says they did not understand or asks for simpler wording.
  Example: "I don't understand", "Explain it simpler."

- greeting:
  Greeting only.
  Example: "hello", "hi", "good morning."

- gratitude:
  Thanks or acknowledgment.
  Example: "thanks", "thank you", "okay thank you."

- small_talk:
  Polite conversation that is not a technical request.
  Example: "how are you?", "nice", "cool."

- out_of_scope_question:
  A meaningful question but unrelated to EMV specifications, payment cards, APDUs,
  terminals, ICC, cryptography, transaction processing, or project documents.
  Example: "Why is football called soccer in the USA?"

- project_question:
  A question about this chatbot project, code, architecture, Docker, LangChain,
  Chroma, Ollama, FastAPI, frontend, or implementation.
  Example: "Why is my router slow?", "How do I stream the answer?"

- frontend_command:
  UI command or user action.
  Example: "clear chat", "new conversation", "reset", "show sources panel."

- empty_input:
  Empty or whitespace-only input.

- noise:
  Random text, gibberish, unclear input, or meaningless input.

Important rules:
- Return only one category label.
- Do NOT rewrite the question.
- Do NOT answer the question.
- If a question is EMV-related and also a definition, comparison, procedure, source,
  citation, or summary request, choose the more specific category.
- If the input depends on previous chat history, choose contextual_follow_up.
- If the user asks about the application implementation, choose project_question.
- If unsure, return noise.

Chat history:
{chat_history}

Latest input:
{question}

Category:
"""


REWRITE_PROMPT = """
You are a query rewriting assistant for an EMV technical RAG chatbot.

Rewrite the user's latest follow-up into a standalone EMV-related question.

Rules:
- Use chat history only to resolve references like "it", "this", "that", "more", "explain more".
- Do not answer the question.
- Do not add information that is not implied by the conversation.
- Do not invent a new topic.
- Return only the rewritten standalone question.

Chat history:
{chat_history}

Latest follow-up:
{question}

Standalone question:
"""


RAG_PROMPT = """
You are an EMV specification assistant.

Answer the user question using ONLY the retrieved context.

Rules:
1. Use only the provided context.
2. Do not invent facts, definitions, steps, or references.
3. If the answer is not clearly present in the context, say exactly:
   "I could not find this in the retrieved EMV sources."
4. Keep the answer technical, clear, and concise.
5. List sources after the answer in this format:
Book | full Section path | Section Title | page number.

Context:
{context}

Question:
{question}

Answer:
"""


def get_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]


def format_history_for_prompt(history: List, max_messages: int = 6) -> str:
    if not history:
        return "No previous conversation."

    recent_history = history[-max_messages:]
    lines = []

    for message in recent_history:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"Assistant: {message.content}")

    return "\n".join(lines)


def normalize_label(label: str) -> InputType:
    label = label.strip().lower()

    # Defensive cleaning in case the model returns extra text.
    label = label.replace("category:", "").strip()
    label = label.split()[0].strip() if label else "noise"

    allowed = {
        "standalone_emv_question",
        "contextual_follow_up",
        "definition_question",
        "comparison_question",
        "procedure_question",
        "source_lookup",
        "citation_request",
        "summarization_request",
        "clarification_request",
        "greeting",
        "gratitude",
        "small_talk",
        "out_of_scope_question",
        "project_question",
        "frontend_command",
        "empty_input",
        "noise",
    }

    if label in allowed:
        return label  # type: ignore

    return "noise"


def classify_input(llm, question: str, history: List) -> InputType:
    if not question or not question.strip():
        return "empty_input"

    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)

    messages = prompt.invoke({
        "chat_history": format_history_for_prompt(history),
        "question": question,
    })

    result = llm.invoke(messages)

    return normalize_label(result.content)


def rewrite_question(llm, question: str, history: List) -> str:
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)

    messages = prompt.invoke({
        "chat_history": format_history_for_prompt(history),
        "question": question,
    })

    rewritten = llm.invoke(messages)

    return rewritten.content.strip()


def format_context(scored_docs, max_chars_per_chunk: int = 1200):
    context_parts = []

    for i, (doc, distance) in enumerate(scored_docs, start=1):
        metadata = doc.metadata

        header = (
            f"[Chunk {i}] "
            f"Book: {metadata.get('doc_id', 'Unknown')} | "
            f"Section: {metadata.get('section_number', 'Unknown')} | "
            f"Title: {metadata.get('title', 'Unknown')} | "
            f"Page: {metadata.get('page_num', 'Unknown')} | "
            f"Distance: {distance:.4f}"
        )

        text = doc.page_content[:max_chars_per_chunk]
        context_parts.append(header + "\n" + text)

    return "\n\n".join(context_parts)


def format_source(metadata: dict):
    return {
        "doc_id": metadata.get("doc_id", "Unknown document"),
        "section_number": metadata.get("section_number", "Unknown section"),
        "title": metadata.get("title", "Unknown title"),
        "page": metadata.get("page_num", "Unknown page"),
    }


def build_empty_response(
    session_id: str,
    question: str,
    input_type: InputType,
    answer: str,
    history_len: int,
    router_time: float,
    total_time: float,
):
    return {
        "session_id": session_id,
        "input_type": input_type,
        "original_question": question,
        "standalone_question": None,
        "answer": answer,
        "sources": [],
        "retrieved_chunks": [],
        "metrics": {
            "history_messages_count": history_len,
            "retrieved_chunks_count": 0,
            "best_distance": None,
            "worst_distance": None,
            "average_distance": None,
            "router_time_seconds": router_time,
            "rewrite_time_seconds": 0,
            "retrieval_time_seconds": 0,
            "generation_time_seconds": 0,
            "total_time_seconds": total_time,
        },
    }


def build_non_rag_answer(input_type: InputType) -> str:
    if input_type == "greeting":
        return "Hello! Ask me an EMV-related question whenever you are ready."

    if input_type == "gratitude":
        return "You're welcome! Ask me another EMV question whenever you want."

    if input_type == "small_talk":
        return "I'm here to help with EMV specifications and your RAG chatbot project."

    if input_type == "empty_input":
        return "Please enter a clear EMV-related question."

    if input_type == "noise":
        return "I could not understand the request. Please ask a clear EMV-related question."

    if input_type == "out_of_scope_question":
        return "I can only answer questions related to the retrieved EMV specification sources."

    if input_type == "project_question":
        return (
            "This question is about the chatbot project implementation, not the EMV vector database. "
            "Handle it with a separate project-assistant route or answer it outside the EMV RAG pipeline."
        )

    if input_type == "frontend_command":
        return "This looks like a UI command. Handle it in the frontend or with a dedicated command endpoint."

    return "I could not process this request."


def should_store_non_rag_message(input_type: InputType) -> bool:
    return input_type in {"greeting", "gratitude", "small_talk"}


def ask_conversational_rag(
    question: str,
    session_id: str = "default",
    k: int = 3,
):
    vectorstore = get_langchain_chroma()
    llm = get_llm()

    history = get_history(session_id)

    total_start = time.perf_counter()

    router_start = time.perf_counter()
    input_type = classify_input(llm, question, history)
    router_end = time.perf_counter()

    if (
        input_type in {
            "greeting",
            "gratitude",
            "small_talk",
            "out_of_scope_question",
            "project_question",
            "frontend_command",
            "empty_input",
            "noise",
        }
    ):
        answer_text = build_non_rag_answer(input_type)

        if should_store_non_rag_message(input_type):
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer_text))

        total_end = time.perf_counter()

        return build_empty_response(
            session_id=session_id,
            question=question,
            input_type=input_type,
            answer=answer_text,
            history_len=len(history),
            router_time=router_end - router_start,
            total_time=total_end - total_start,
        )

    rewrite_start = time.perf_counter()

    if input_type == "contextual_follow_up":
        standalone_question = rewrite_question(llm, question, history)
    elif input_type in RAG_INPUT_TYPES:
        standalone_question = question
    else:
        standalone_question = question

    rewrite_end = time.perf_counter()

    retrieval_start = time.perf_counter()

    scored_docs = vectorstore.similarity_search_with_score(
        standalone_question,
        k=k,
    )

    retrieval_end = time.perf_counter()

    context = format_context(scored_docs)

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    generation_start = time.perf_counter()

    messages = prompt.invoke({
        "context": context,
        "question": standalone_question,
    })

    response_chunks = llm.stream(messages)

    full_answer = ""

    for chunk in response_chunks:
        token = chunk.content or ""
        print(token, end="", flush=True)
        full_answer += token

    generation_end = time.perf_counter()
    total_end = time.perf_counter()

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=full_answer))

    sources = []
    seen = set()
    retrieved_chunks = []

    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        source = format_source(doc.metadata)

        source_key = (
            source["doc_id"],
            source["section_number"],
            source["title"],
            source["page"],
        )

        if source_key not in seen:
            sources.append(source)
            seen.add(source_key)

        retrieved_chunks.append({
            "rank": rank,
            "distance": distance,
            "text_preview": doc.page_content[:500],
            "metadata": doc.metadata,
        })

    distances = [distance for _, distance in scored_docs]

    return {
        "session_id": session_id,
        "input_type": input_type,
        "original_question": question,
        "standalone_question": standalone_question,
        "answer": full_answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "metrics": {
            "history_messages_count": len(history),
            "retrieved_chunks_count": len(scored_docs),
            "best_distance": min(distances) if distances else None,
            "worst_distance": max(distances) if distances else None,
            "average_distance": sum(distances) / len(distances) if distances else None,
            "router_time_seconds": router_end - router_start,
            "rewrite_time_seconds": rewrite_end - rewrite_start,
            "retrieval_time_seconds": retrieval_end - retrieval_start,
            "generation_time_seconds": generation_end - generation_start,
            "total_time_seconds": total_end - total_start,
        },
    }


def clear_conversation(session_id: str = "default"):
    chat_histories[session_id] = []


def stream_conversational_rag(question: str, session_id: str = "default", k: int = 3):
    vectorstore = get_langchain_chroma()
    llm = get_llm()
    history = get_history(session_id)

    input_type = classify_input(llm, question, history)

    if (
        input_type in {
            "greeting",
            "gratitude",
            "small_talk",
            "out_of_scope_question",
            "project_question",
            "frontend_command",
            "empty_input",
            "noise",
        }
    ):
        answer = build_non_rag_answer(input_type)

        if should_store_non_rag_message(input_type):
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer))

        yield answer
        return

    if input_type == "contextual_follow_up":
        standalone_question = rewrite_question(llm, question, history)
    else:
        standalone_question = question

    scored_docs = vectorstore.similarity_search_with_score(
        standalone_question,
        k=k,
    )

    context = format_context(scored_docs)

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    messages = prompt.invoke({
        "context": context,
        "question": standalone_question,
    })

    full_answer = ""

    for chunk in llm.stream(messages):
        token = chunk.content or ""
        full_answer += token
        yield token

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=full_answer))
