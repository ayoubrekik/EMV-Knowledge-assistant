from typing import Literal, Tuple
from langchain_core.documents import Document

InputType = Literal[
    "emv_question",
    "hex_decode_question",
    "definition_question",
    "comparison_question",
    "contextual_follow_up",
    "source_lookup",
    "tag_lookup_question",
    "noise",
]

ScoredDoc = Tuple[Document, float]
