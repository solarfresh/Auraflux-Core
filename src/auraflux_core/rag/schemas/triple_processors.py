from typing import List

from pydantic import BaseModel, Field

from auraflux_core.rag.schemas.chunker import TripleItem


class FlaggedTripleItem(TripleItem):
    flag_reasons: List[str] = Field(
        default_factory=list,
        alias="_flag_reasons",
        description="List of specific rule violation reasons identified during inspection."
    )

class TripleRuleCheckerOutput(BaseModel):
    clean_triples: List[TripleItem]
    flagged_triples: List[FlaggedTripleItem]
    has_flagged: bool
    total_count: int
    flagged_count: int
