from typing import Any, Dict, List

from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    llm_config: Dict[str, Any]
    system_messages: List[Dict[str, str]] | None = None

    @property
    def model(self) -> str:
        # Assumes the first model in the config_list is the primary one
        config_list = self.llm_config.get("config_list", []) if self.llm_config else []
        return config_list[0].get("model") if config_list else ""
