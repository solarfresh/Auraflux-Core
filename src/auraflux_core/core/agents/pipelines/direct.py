import json
from typing import TYPE_CHECKING, Any, Dict, List

from auraflux_core.core.agents.pipelines.base import (BaseAgentPipeline,
                                                      PipelineRegistry)
from auraflux_core.core.schemas.messages import Message

if TYPE_CHECKING:
    from auraflux_core.core.agents.base import BaseAgent


@PipelineRegistry.register("direct")
class DirectPipeline(BaseAgentPipeline):
    """
    Default fallback pipeline for direct single-turn LLM generation.
    Completely decoupled and universal with zero handler requirements.
    """

    def get_execute_pipeline(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Declares a single direct execution step for Strategy dispatching.
        """
        return [
            {"method_name": "_execute_pipeline", "kwargs": {"payload": payload}}
        ]

    async def _execute_pipeline(self, agent: "BaseAgent", payload: Dict[str, Any], **kwargs) -> Any:
        if "messages" in payload and isinstance(payload["messages"], list):
            messages = payload["messages"]
        elif "prompt" in payload:
            messages = [Message(role="user", content=str(payload["prompt"]), name=agent.name)]
        else:
            content = json.dumps(payload, ensure_ascii=False)
            messages = [Message(role="user", content=content, name=agent.name)]

        response = await agent.generate(messages)

        return response.content
