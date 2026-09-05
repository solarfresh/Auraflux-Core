import json
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional

from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.clients import LLMRequest, LLMResponse
from auraflux_core.core.schemas.messages import Message
from auraflux_core.core.schemas.tools import (ToolCallProtocol,
                                              ToolExecutionStrategy)
from auraflux_core.core.tools import ToolExecutor


class BaseAgent(ABC):
    """
    Base class for all agents in the Auraflux system, using AutoGen's ConversableAgent as the foundation.

    This class provides a shared logging setup and a consistent initialization pattern.
    The agent's specific behavior should be defined in subclasses by implementing their
    role within an AutoGen GroupChat or other conversational flows.
    """
    def __init__(self, config: AgentConfig, client_manager: ClientManager):
        self.config = config
        self.client_manager = client_manager
        self.logger = setup_logging(name=f"[{self.config.name}]")
        self.logger.info(f"Agent '{self.config.name}' initialized.")
        if config.system_message is not None:
            self.system_message = config.system_message
        else:
            self.system_message = str(self._message_mapper(self.get_system_message_map()))

        self.tool_executor = ToolExecutor(
            tools=self.config.tools or [],
            tool_call_protocol=self.config.tool_call_protocol
        )

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def name(self) -> str:
        return self.config.name

    async def generate(self, messages: List[Message], tool_args_map: Dict[str, Any] | None = None) -> Message:
        copied_messages = [deepcopy(msg) for msg in messages[-self.config.turn_limit:]]

        try:
            if self.config.tool_execution_strategy == 'DIRECT':
                tool_output_message = await self.generate_tool_message(copied_messages, tool_args_map=tool_args_map)
                return tool_output_message

            if self.config.tool_execution_strategy == 'REFLECTIVE':
                last_message = await self.generate_tool_message(copied_messages, tool_args_map=tool_args_map)
                self.logger.debug(f"Tool output: {last_message.content}")
                copied_messages.append(last_message)

            return await self.generate_llm_message(copied_messages)
        except Exception as e:
            self.logger.error(f"Error during agent generation for agent '{self.name}': {e}")
            return Message(role='assistant', content="Error: Could not generate a response.", name=self.name)

    async def generate_llm_message(self, messages: List[Message]) -> Message:
        last_message = messages[-1]

        try:
            cot_to_append = self.config.cot_message or self._message_mapper(self.get_cot_message_map())
            if cot_to_append:
                last_message.content += f"\n\n{cot_to_append}"

            request = LLMRequest(
                provider=self.provider,
                model=self.model,
                messages=messages,
                system_message=self.system_message,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                thinking_level=self.config.thinking_level,
            )

            self.logger.debug(f"Sending request to LLM: {request}")
            response: LLMResponse = await self.client_manager.generate(request)
            self.logger.debug(f"Received response from LLM: {response}")

            output_string = self.postprocess_llm_output(response.text)

            return Message(role='assistant', content=output_string, name=self.name, token_usage=response.token_usage)
        except Exception as e:
            self.logger.error(f"Error during LLM generation for agent '{self.name}': {e}")
            raise e

    async def generate_tool_message(self, messages: List[Message], tool_args_map: Dict[str, Any] | None = None) -> Message:
        self.logger.info("Generating tool message...")
        tool_call_data = await self._decide_tool_calls(messages)
        message = await self._execute_tool_calls(tool_call_data, tool_args_map)
        return message

    def generate_stream(self, message: Message, chat_history: List[Message]) -> Generator[Message, Any, Any]:

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
        """
        Abstract method to be implemented by subclasses to provide a mapping of model families
        to their respective system messages.
        """
        pass

    def get_cot_message_map(self) -> Dict[str, str] | None:
        """
        Method to be optionally overridden by subclasses to provide a mapping of model families
        to their respective chain-of-thought (CoT) messages.
        """
        return None

    def get_tool_call(self, messages: List[Message]) -> Dict[str, Any]:
        return {}

    def get_tool_message_map(self) -> Dict[str, str] | None:
        """
        Method to be optionally overridden by subclasses to provide a mapping of model families
        to their respective tool-use messages.
        """
        return None

    async def _decide_tool_calls(self, messages: List[Message]) -> Dict[str, Any]:
        tool_call_data = {}
        tool_message = self._message_mapper(self.get_tool_message_map())

        if self.config.tool_call_protocol == ToolCallProtocol.PROMPT.value:
            if tool_message is None:
                self.logger.warning(f"Tool call protocol is set to PROMPT but no tool message is defined for agent '{self.name}'. Proceeding without tool call.")
                return {}

            request = LLMRequest(
                provider=self.provider,
                model=self.model,
                messages=messages,
                system_message=tool_message,
                thinking_level=self.config.thinking_level
            )
            self.logger.debug(f"Sending request to LLM: {request}")
            response: LLMResponse = await self.client_manager.generate(request)
            self.logger.debug(f"Received response from LLM: {response}")
            tool_call_data = self.postprocess_tool_output(response.text)
        elif self.config.tool_call_protocol == ToolCallProtocol.NATIVE.value:
            formatted_tools = self.tool_executor.convert_tool_specs(self.provider)
            request = LLMRequest(
                provider=self.provider,
                model=self.model,
                messages=messages,
                system_message=self.system_message,
                thinking_level=self.config.thinking_level,
                tools=formatted_tools
            )
            response: LLMResponse = await self.client_manager.generate(request)

            if response.tool_calls:
                first_call = response.tool_calls[0] if isinstance(response.tool_calls, list) else response.tool_calls
                tool_call_data = {
                    "tool": getattr(first_call, "name", first_call.get("name") if isinstance(first_call, dict) else None),
                    "args": getattr(first_call, "arguments", first_call.get("args", {}) if isinstance(first_call, dict) else {})
                }
            else:
                raise ValueError("NATIVE protocol expected a tool call but got text response.")
        else:
            tool_call_data = self.get_tool_call(messages=messages)

        return tool_call_data

    async def _execute_tool_calls(self, tool_call_data: Dict[str, Any], tool_args_map: Dict[str, Any] | None = None) -> Message:
        tool_name = tool_call_data.get('tool', 'default')
        tool_call_args = tool_call_data.get('args', {})

        self.logger.debug(f"Delegating execution of tool '{tool_name}' to ToolExecutor.")
        return await self.tool_executor.run(
            tool_name=tool_name,
            tool_args=tool_call_args,
            tool_args_override_map=tool_args_map
        )

    def _message_mapper(self, msg_map: Dict[str, str] | None) -> str | None:
        if msg_map is None:
            return None

        return msg_map.get(self.config.lang, 'default')

    def postprocess_tool_output(self, output_string: str) -> Any:
        json_object = self._parse_json_output(output_string)
        return json_object

    def postprocess_llm_output(self, output_string: str) -> str:
        if self.config.output_format == 'JSON':
            json_object = self._parse_json_output(output_string)
            return json.dumps(json_object, ensure_ascii=False)

        return output_string

    def _parse_json_output(self, output_string: str) -> Dict:
        json_pattern = r"```json\s*(\{.*\})\s*```"
        match = re.search(json_pattern, output_string, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            try:
                return json.loads(output_string)
            except Exception as e:
                self.logger.warning(output_string)
                raise e

        clean_string = re.sub(r'\\\w+\{([^}]+)\}', r'->(\1)->', json_string)
        clean_string = clean_string.replace('$', '')

        return json.loads(clean_string)
