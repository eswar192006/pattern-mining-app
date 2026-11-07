"""What the language model must return, and the schema it is shown.

    {"intent": "mining_request", "spec": {...}, "message": null}
    {"intent": "out_of_scope",   "spec": null,  "message": "I can only help with ..."}

Whatever comes back is validated by pydantic here and then by validate_spec(), so a
wrong or odd answer from the model can only ever become an error or a question.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.spec import Spec


class LLMReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["mining_request", "out_of_scope"] = Field(
        description="mining_request if the question can be answered by mining the sales data "
                    "(even if details are missing), otherwise out_of_scope.")
    spec: Spec | None = Field(None, description="The request. null when intent is out_of_scope.")
    message: str | None = Field(None, description="One short friendly sentence, only for out_of_scope.")

    @model_validator(mode="after")
    def _mining_request_always_has_a_spec(self):
        if self.intent == "mining_request" and self.spec is None:
            self.spec = Spec()          # nothing specified: the validator will ask what the user wants
        return self


# Gemini accepts a subset of JSON Schema: inline $refs, and drop keywords it may reject.
_DROP = {"$defs", "default", "title", "exclusiveMinimum", "exclusiveMaximum", "minimum", "maximum"}


def _inline(node, defs):
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline(defs[node["$ref"].split("/")[-1]], defs)
        return {k: _inline(v, defs) for k, v in node.items() if k not in _DROP}
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def simplified_schema(model) -> dict:
    """A pydantic model's JSON schema in the simple form sent to the model (no $ref, no numeric
    bounds). Bounds are enforced afterwards by pydantic, so nothing is lost."""
    raw = model.model_json_schema()
    return _inline(raw, raw.get("$defs", {}))


def llm_json_schema() -> dict:
    return simplified_schema(LLMReply)


class FollowUpReply(BaseModel):
    """What the model returns for a follow-up message: an EDIT of the current request, not a new one.
    Only the fields that change are given; everything else stays exactly as it was."""
    model_config = ConfigDict(extra="forbid")

    intent: Literal["mining_request", "out_of_scope", "reset", "undo"] = Field(
        description="mining_request: change or replace the current request. reset: start over. "
                    "undo: go back to the previous request. out_of_scope: not about the sales patterns.")
    fresh: bool = Field(False, description="true only if the message is a NEW question unrelated to the current "
                                           "request; then spec must be complete.")
    spec: Spec | None = Field(None, description="ONLY the fields that change (complete when fresh is true).")
    clear: list[str] = Field(default_factory=list, description="Settings to remove completely, "
                                                               "e.g. filters.weekdays or min_support.")
    message: str | None = Field(None, description="One short friendly sentence, only for out_of_scope.")


def followup_json_schema() -> dict:
    return simplified_schema(FollowUpReply)
