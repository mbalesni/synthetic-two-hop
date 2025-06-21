from typing import Any, Literal

DEFAULT_SYSTEM_MESSAGE = "Answer the following question."
NO_COT_SYSTEM_MESSAGE = "Answer the following questions directly, without any other text before or after your answer."  # TODO(mbalesni): actually evaluate with this. I changed this after starting the run.
COT_SYSTEM_MESSAGE = "Answer the following questions step by step."


AuxLossType = Literal["logit", "embed_cosine", "embed_mse", "collected_rep_cosine"]
