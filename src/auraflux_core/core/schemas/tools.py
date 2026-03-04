from typing import Dict

from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    args: Dict = Field(default={})
