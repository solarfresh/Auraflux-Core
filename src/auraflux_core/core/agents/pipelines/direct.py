import json
from typing import TYPE_CHECKING, Any, Dict

from auraflux_core.core.agents.pipelines.base import (BaseAgentPipeline,
                                                      PipelineRegistry)
from auraflux_core.core.schemas.messages import Message

if TYPE_CHECKING:
    from auraflux_core.core.agents.base_agent import BaseAgent


@PipelineRegistry.register("direct")
class DirectPipeline(BaseAgentPipeline):
    """
    Default fallback pipeline for direct single-turn LLM generation.
    Completely decoupled and universal with zero handler requirements.
    """

    async def execute(self, agent: "BaseAgent", payload: Dict[str, Any]) -> Any:
        if "messages" in payload and isinstance(payload["messages"], list):
            messages = payload["messages"]
        elif "prompt" in payload:
            messages = [Message(role="user", content=str(payload["prompt"]), name=agent.name)]
        else:
            content = json.dumps(payload, ensure_ascii=False)
            messages = [Message(role="user", content=content, name=agent.name)]

        response = await agent.generate(messages)

        return response.content
