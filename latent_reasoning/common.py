from typing import Any, Literal

DEFAULT_SYSTEM_MESSAGE = "Answer the following question."
NO_COT_SYSTEM_MESSAGE = "Answer the following questions directly, without any other text before or after your answer."  # TODO(mbalesni): actually evaluate with this. I changed this after starting the run.
COT_SYSTEM_MESSAGE = "Answer the following questions step by step."


AuxLossType = Literal["logit", "embed_cosine", "embed_mse", "collected_rep_cosine"]


def merge_system_message(element: dict[str, Any]) -> dict[str, Any]:
    messages = element["messages"]
    # Combine system message with the first user message
    system_content = messages[0]["content"]
    messages = messages[1:]  # Remove system message
    if messages[0]["role"] == "user":
        messages[0]["content"] = f"{system_content}\n\n{messages[0]['content']}"
    element["messages"] = messages
    return element
