import os
import math
import time
from uuid import UUID
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.core.rag.emv_registry_decoder import decode_from_registry
from src.core.rag.hex_preprocessor import get_hex_breakdown, inject_hex_breakdown
from src.core.rag.langchain_chroma import get_langchain_chroma
from src.core.rag.langchain_llm import get_llm
from src.core.rag.tag_lookup import (
    extract_emv_tag,
    build_tag_lookup_context
)

from .context_formatter import format_context
from .debug_utils import save_final_context_to_txt
from .generation import build_non_rag_answer, stream_answer
from .hex_utils import extract_active_bits, extract_matching_bit_definitions
from .history import clear_conversation, get_history
from .persistence import (
    create_and_save_rag_metadata,
    create_assistant_message,
    create_user_message,
    get_or_create_chat_session,
    save_rag_sources,
    touch_session,
    relevance_percent,
)
from .retrieval import retrieve_documents
from .routing import has_direct_definition_match, is_definition_question, rewrite_for_retrieval,  classify_input
from .source_utils import build_sources_and_chunks, citation_from_source, source_from_doc
from .streaming import sse_event

load_dotenv()


def _metadata_payload(
    session_id,
    user_id,
    input_type,
    question,
    standalone_question,
    answer,
    sources,
    retrieved_chunks,
    hex_breakdown,
    history,
    scored_docs,
    best_relevance,
    worst_relevance,
    average_relevance,
    router_time,
    rewrite_time,
    retrieval_time,
    generation_time,
    total_time,
):
    return {
        "session_id": session_id,
        "user_id": user_id,
        "input_type": input_type,
        "original_question": question,
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "hex_breakdown": hex_breakdown,
        "metrics": {
            "history_messages_count": len(history),
            "retrieved_chunks_count": len(scored_docs),
            "best_relevance": best_relevance,
            "average_relevance": average_relevance,
            "worst_relevance": worst_relevance,
            "router_time_seconds": router_time,
            "rewrite_time_seconds": rewrite_time,
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": generation_time,
            "total_time_seconds": total_time,
        },
    }


def stream_conversational_rag(db, question: str, session_id: str = "default", user_id: str | None = None, k: int = 7,temp=0):
    if not user_id:
        raise ValueError("user_id is required")


    yield sse_event("status", {
    "title": "Preparing session",
    "message": "Opening the current chat session..."
    })
    user_uuid = UUID(user_id)
    chat_session = get_or_create_chat_session(
        db=db,
        session_id=session_id,
        user_id=user_id,
        question=question,
    )

    total_start = time.perf_counter()


    yield sse_event("status", {
        "title": "Loading knowledge base",
        "message": "Connecting to ChromaDB and loading the language model..."
    })
    vectorstore = get_langchain_chroma()
    llm = get_llm(temp)
    history = get_history(session_id)


    yield sse_event("status", {
        "title": "Understanding question",
        "message": "Detecting whether this is an EMV question, source lookup, hex decoding, or casual input..."
    })
    router_start = time.perf_counter()
    input_type = classify_input(llm, question, history)
    print(f"Input type classified as: {input_type}")
    print("Okay Okay, let's see what we can do with this question...")
    #input_type = "tag_lookup_question"
    router_end = time.perf_counter()

    user_message = create_user_message(
        db=db,
        chat_session=chat_session,
        user_uuid=user_uuid,
        question=question,
        input_type=input_type,
    )
    if input_type == "noise":
        generation_start = time.perf_counter()
        answer = build_non_rag_answer(llm, question)
        generation_end = time.perf_counter()
        total_end = time.perf_counter()

        assistant_message = create_assistant_message(db, chat_session, user_uuid, answer)
        create_and_save_rag_metadata(
            db=db,
            chat_session=chat_session,
            user_uuid=user_uuid,
            user_message=user_message,
            assistant_message=assistant_message,
            question=question,
            standalone_question=None,
            input_type=input_type,
            scored_docs=[],
            router_time=router_end - router_start,
            retrieval_time=0,
            generation_time=generation_end - generation_start,
            total_time=total_end - total_start,
            model_name="qwen3:8b" ,#os.getenv("OLLAMA_MODEL") or "qwen3:8b",
            embedding_model="none",
        )
        touch_session(chat_session)
        db.commit()

        yield sse_event("token", answer)
        yield sse_event("metadata", _metadata_payload(
            session_id, user_id, input_type, question, None, answer, [], [], None, history,
            [], None, None, None, router_end - router_start, 0, 0,
            generation_end - generation_start, total_end - total_start,
        ))
        yield sse_event("done", "[DONE]")
        return

    if input_type in {
        "hex_decode_question",
        "source_lookup",
        # "emv_question",
        "document_question",
        "tag_lookup_question",
        "definition_question",
        "comparison_question",
    }:
        retrieval_query = question
        rewrite_time = 0.0
    else:
        rewrite_start = time.perf_counter()

        retrieval_query = rewrite_for_retrieval(
            llm=llm,
            question=question,
            history=history,
            current_topic=chat_session.current_topic,
        )
        print(f"Retrieval query rewritten as: {retrieval_query}")

        rewrite_end = time.perf_counter()
        rewrite_time = rewrite_end - rewrite_start
    # Update the topic only for real RAG questions
    if input_type in {
        # "emv_question",
        "document_question",
        "hex_decode_question",
        "source_lookup",
        "tag_lookup_question",
        "definition_question",
        "comparison_question",
    }:
        chat_session.current_topic = retrieval_query
    if input_type == "hex_decode_question":
        registry_answer = decode_from_registry(retrieval_query)
        if registry_answer:
            generation_start = time.perf_counter()
            generation_end = time.perf_counter()
            total_end = time.perf_counter()

            assistant_message = create_assistant_message(db, chat_session, user_uuid, registry_answer)
            create_and_save_rag_metadata(
                db=db,
                chat_session=chat_session,
                user_uuid=user_uuid,
                user_message=user_message,
                assistant_message=assistant_message,
                question=question,
                standalone_question=retrieval_query,
                input_type=input_type,
                scored_docs=[],
                router_time=router_end - router_start,
                retrieval_time=0,
                generation_time=0,
                total_time=total_end - total_start,
                model_name="local-registry",
                embedding_model="none",
            )
            touch_session(chat_session)
            db.commit()

            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=registry_answer))

            yield sse_event("token", registry_answer)
            yield sse_event("metadata", _metadata_payload(
                session_id, user_id, input_type, question, retrieval_query, registry_answer,
                [{"doc_id": "emv_decode_registry.json"}], [], None, history,
                [], None, None, None, router_end - router_start, rewrite_time, 0, 0,
                total_end - total_start,
            ))
            yield sse_event("done", "[DONE]")
            return


    retrieval_start = time.perf_counter()
    yield sse_event("status", {
        "title": "Retrieving context",
        "message": "Searching the vector database for relevant EMV chunks..."
    })

    scored_docs = retrieve_documents(
        vectorstore=vectorstore,
        query=retrieval_query,
        final_k=k,
        semantic_k=20,
        bm25_k=15,
    )

    retrieval_end = time.perf_counter()

    yield sse_event("status", {
        "title": "Preparing context",
        "message": "Formatting retrieved chunks for the model..."
    })
    if input_type == "tag_lookup_question":
        tag = extract_emv_tag(question)
        context = build_tag_lookup_context(scored_docs, tag)
    else:
        context = format_context(scored_docs, query=retrieval_query)

    if input_type == "hex_decode_question":
        hex_breakdown = get_hex_breakdown(retrieval_query)
        context_with_hex = inject_hex_breakdown(retrieval_query, context)
        active_bits = extract_active_bits(hex_breakdown)
        matched_definitions = extract_matching_bit_definitions(context=context_with_hex, active_bits=active_bits)

        decode_sources = []
        for doc, _ in scored_docs:
            citation = citation_from_source(source_from_doc(doc))
            if citation not in decode_sources:
                decode_sources.append(citation)

        context = f"""
HEX BREAKDOWN:
{hex_breakdown}

MATCHED ACTIVE BIT DEFINITIONS:
{matched_definitions}

SOURCES:
{chr(10).join(decode_sources)}
""".strip()
    else:
        hex_breakdown = None

    save_final_context_to_txt(
        filename="final_context_sent_to_llm.txt",
        context=context,
        query=retrieval_query,
        input_type=input_type,
    )

    generation_start = time.perf_counter()
    full_answer = ""
    
    if (input_type == "contextual_follow_up"):
        input_type = "document_question"

    for chunk in stream_answer(llm, context, retrieval_query, input_type, history):
        token = chunk.content or ""
        full_answer += token
        yield sse_event("token", token)

    generation_end = time.perf_counter()
    total_end = time.perf_counter()

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=full_answer))

    sources, retrieved_chunks = build_sources_and_chunks(scored_docs)

    scores = [
        doc.metadata.get("cross_encoder_score")
        for doc, _ in scored_docs
        if doc.metadata.get("cross_encoder_score") is not None
    ]

    best_score = max(scores) if scores else None
    worst_score = min(scores) if scores else None
    average_score = sum(scores) / len(scores) if scores else None

    best_relevance = relevance_percent(best_score)
    average_relevance = relevance_percent(average_score)
    worst_relevance = relevance_percent(worst_score)
    

    assistant_message = create_assistant_message(db, chat_session, user_uuid, full_answer)
    rag_metadata, _, _, _ = create_and_save_rag_metadata(
        db=db,
        chat_session=chat_session,
        user_uuid=user_uuid,
        user_message=user_message,
        assistant_message=assistant_message,
        question=question,
        standalone_question=retrieval_query,
        input_type=input_type,
        scored_docs=scored_docs,
        router_time=router_end - router_start,
        retrieval_time=retrieval_end - retrieval_start,
        generation_time=generation_end - generation_start,
        total_time=total_end - total_start,
        model_name= "qwen3:8b" ,#os.getenv("OLLAMA_MODEL"),
        embedding_model=os.getenv("EMBEDDING_MODEL"),
    )
    save_rag_sources(db, rag_metadata.id, scored_docs)
    touch_session(chat_session)
    db.commit()

    yield sse_event("metadata", _metadata_payload(
        session_id, user_id, input_type, question, retrieval_query, full_answer,
        sources, retrieved_chunks, hex_breakdown, history, scored_docs,
        best_relevance, worst_relevance, average_relevance,
        router_end - router_start, rewrite_time,
        retrieval_end - retrieval_start, generation_end - generation_start,
        total_end - total_start,
    ))
    yield sse_event("done", "[DONE]")
