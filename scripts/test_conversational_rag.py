from src.core.rag.conversational_rag_service import (
    ask_conversational_rag,
    clear_conversation,
)

def fmt_float(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

def print_result(result):
    print("\n=== Original Question ===")
    print(result["original_question"])

    print("\n=== Rewritten Standalone Question ===")
    print(result["standalone_question"])

    print("\n=== Answer ===")
    print(result["answer"])

    print("\n=== Sources ===")
    for i, source in enumerate(result["sources"], start=1):
        print(
            f"- [{i}] {source['doc_id']} | "
            f"Section {source['section_number']} | "
            f"{source['title']} | page {source['page']}"
        )

    print("\n=== Metrics ===")
    metrics = result["metrics"]

    print(f"History messages: {fmt_float(metrics['history_messages_count'])}")
    print(f"Retrieved chunks: {fmt_float(metrics['retrieved_chunks_count'])}")
    print(f"Best distance: {fmt_float(metrics['best_distance'])}")
    print(f"Worst distance: {fmt_float(metrics['worst_distance'])}")
    print(f"Average distance: {fmt_float(metrics['average_distance'])}")
    print(f"Router time: {metrics.get('router_time_seconds'):.4f}s")
    print(f"Rewrite time: {fmt_float(metrics['rewrite_time_seconds'])}s")
    print(f"Retrieval time: {fmt_float(metrics['retrieval_time_seconds'])}s")
    print(f"Generation time: {fmt_float(metrics['generation_time_seconds'])}s")
    print(f"Total time: {fmt_float(metrics['total_time_seconds'])}s")
def main():
    session_id = "new-session"

    clear_conversation(session_id)
    first_question = "Could you please explain more about Signed Static Application Data (SDA)?"
    second_question = "Ok thank you "
    third_question = "i didnt understand"

    result1 = ask_conversational_rag(
        question=first_question,
        session_id=session_id,
        k=3
    )

    print("\n\n######## FIRST QUESTION ########")
    print_result(result1)

    result2 = ask_conversational_rag(
        question=second_question,
        session_id=session_id,
        k=3
    )

    print("\n\n######## FOLLOW-UP QUESTION ########")
    
    print_result(result2)
    
    # result3 = ask_conversational_rag(
    #     question=third_question,
    #     session_id=session_id,
    #     k=3
    # )

    # print("\n\n######## FOLLOW-UP 3 QUESTION ########")
    # print_result(result3)


if __name__ == "__main__":
    main()