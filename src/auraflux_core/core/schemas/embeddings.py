from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, PositiveInt


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Default Embedding"
    provider: str
    model: str
    dimensions: Optional[PositiveInt] = None
    chunk_size: PositiveInt = 1000
    chunk_overlap: int = 200
    batch_size: PositiveInt = 100
    parameters: Dict[str, Any] = {}
