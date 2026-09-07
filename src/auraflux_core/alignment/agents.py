from abc import ABC, abstractmethod
from typing import Any

from auraflux_core.core.agents.base_agent import BaseAgent


class BaseAlignmentAgent(BaseAgent, ABC):
    @abstractmethod
    async def diagnose_and_verify(self, proposition_id: str, claim_text: str) -> Any:
        pass
