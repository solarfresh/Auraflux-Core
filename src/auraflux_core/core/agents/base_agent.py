import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Generator, List

from auraflux_core.core.agents.pipelines.base import (BaseAgentPipeline,
                                                      PipelineRegistry)
from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.messages import PromptFormatter
from auraflux_core.core.parsers import OutputParser
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.clients import LLMRequest, LLMResponse
from auraflux_core.core.schemas.messages import Message
from auraflux_core.core.tools import ToolExecutor


class BaseAgent(ABC):
    """
    Base class for all agents in the Auraflux system.

    Provides infrastructure capabilities (LLM clients, PromptFormatter, OutputParser, ToolExecutor)
    and delegates execution flow control to a stateless BasePipeline strategy.
    """
    def __init__(self, config: AgentConfig, client_manager: ClientManager):
        self.config = config
        self.client_manager = client_manager
        self.logger = setup_logging(name=f"[{self.config.name}]")
        self.logger.info(f"Agent '{self.config.name}' initialized.")

        self.prompt_formatter = PromptFormatter(
            config=self.config,
            system_message_map=self.get_system_message_map(),
        )

        self.tool_executor = ToolExecutor(
            tools=self.config.tools or [],
            tool_call_protocol=self.config.tool_call_protocol
        )

        self.output_parser = OutputParser()

        pipeline_name = getattr(self.config, "pipeline_name", "direct")
        self.pipeline: BaseAgentPipeline = PipelineRegistry.get(pipeline_name)

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def system_message(self) -> str:
        """Dynamically format the system message using PromptFormatter."""
        return self.prompt_formatter.format_system_message()

    async def generate(self, messages: List[Message]) -> Message:
        """Pure LLM inference execution capability."""
        copied_messages = [deepcopy(msg) for msg in messages]

        try:
            request = LLMRequest(
                provider=self.provider,
                model=self.model,
                messages=copied_messages,
                system_message=self.system_message,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                thinking_level=self.config.thinking_level,
            )

            self.logger.debug(f"Sending request to LLM: {request}")
            response: LLMResponse = await self.client_manager.generate(request)
            self.logger.debug(f"Received response from LLM: {response}")

            output_string = self.postprocess_output(response.text)

            return Message(
                role='assistant',
                content=output_string,
                name=self.name,
                token_usage=response.token_usage
            )
        except Exception as e:
            self.logger.error(f"Error during LLM generation for agent '{self.name}': {e}")
            raise e

    def generate_stream(self, message: Message, chat_history: List[Message]) -> Generator[Message, Any, Any]:
        """Supports streaming response generation."""
        messages = [deepcopy(msg) for msg in chat_history]
        messages.append(message)

        request = LLMRequest(
            provider=self.provider,
            model=self.model,
            messages=messages,
            system_message=self.system_message,
        )
        response_stream = self.client_manager.generate_stream(request)
        for response in response_stream:
            yield Message(role='assistant', content=response.text, name=self.name)

    @abstractmethod
    def get_system_message_map(self) -> Dict[str, str]:
        """Abstract method to be implemented by subclasses to provide system messages."""
        pass

    def register_tools(self, tools: Any) -> "BaseAgent":
        """Delegates tool registration directly to the underlying ToolExecutor."""
        if self.tool_executor:
            self.tool_executor.register_tools(tools)
            self.logger.info(
                f"Updated tools via ToolExecutor. Active tools: {list(self.tool_executor.tool_registry.keys())}"
            )
        return self

    def postprocess_output(self, output_string: str) -> str:
        """Post-processes raw LLM text based on configuration format."""
        if self.config.output_format == 'JSON':
            json_object = self.output_parser.parse_json(output_string)
            return json.dumps(json_object, ensure_ascii=False)

        return self.output_parser.strip_thinking_tags(output_string)

    async def run(self, payload: Dict[str, Any]) -> Any:
        """
        Main execution entry point.
        Delegates control flow execution directly to the bound Pipeline strategy.
        """
        return await self.pipeline.execute(agent=self, payload=payload)
