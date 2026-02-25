from uuid import uuid4
from enum import Enum
from typing import Optional, List, Union
from pydantic import BaseModel, Field


class ConceptualNodeType(str, Enum):
    FOCUS = "FOCUS"
    RESOURCE = "RESOURCE"
    CONCEPT = "CONCEPT"
    INSIGHT = "INSIGHT"
    QUERY = "QUERY"
    NAVIGATION = "NAVIGATION"
    GROUP = "GROUP"


class ConceptualNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    type: ConceptualNodeType
    position: Optional[dict] = None # {"x": float, "y": float}

    # Metadata for AI reasoning/UI feedback
    rationale: Optional[str] = None
