import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from base_tool import BaseTool, ToolSpecConverter

from auraflux_core.core.schemas.messages import Message
from auraflux_core.core.schemas.tools import ToolCallProtocol


class BaseToolExecutor(ABC):
    """
    Abstract base class defining the contract for tool management,
    specification conversion, and asynchronous execution.
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        tool_call_protocol: str = ToolCallProtocol.NATIVE.value,
    ) -> None:
        """
        Initialize the base tool executor.

        Args:
            tools (Optional[List[BaseTool]]): List of configured tool instances.
            tool_call_protocol (str): Protocol used for tool calling (e.g., NATIVE, PROMPT).
        """
        self.tool_call_protocol = tool_call_protocol
        self.tool_registry: Dict[str, BaseTool] = {
            tool.get_name(): tool for tool in (tools or [])
        }

    @abstractmethod
    def convert_tool_specs(self, provider: str) -> List[Any]:
        """Convert registered tools into LLM provider-specific tool schemas/objects."""
        pass

    @abstractmethod
    def get_prompt_text(self) -> str:
        """Convert registered tools into a plain text description for PROMPT mode."""
        pass

    @abstractmethod
    async def run(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_args_override_map: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Asynchronously run a tool by name and return a Message containing the execution result.

        Args:
            tool_name (str): Target tool identifier.
            tool_args (Dict[str, Any]): Arguments passed to the tool.
            tool_args_override_map (Optional[Dict[str, Any]]): Static overrides or supplemental arguments.

        Returns:
            Message: Execution result packaged into a Message with role='tool'.
        """
        pass


class ToolExecutor(BaseToolExecutor):
    """
    Concrete implementation of BaseToolExecutor utilizing ToolSpecConverter.
    """

    def convert_tool_specs(self, provider: str) -> List[Any]:
        """Converts tools strictly into SDK objects or schemas."""
        provider_name = provider.lower()
        if "openai" in provider_name:
            return [
                ToolSpecConverter.to_openai(tool)
                for tool in self.tool_registry.values()
            ]
        elif "gemini" in provider_name or "google" in provider_name:
            return [
                ToolSpecConverter.to_gemini(tool)
                for tool in self.tool_registry.values()
            ]

        return []

    def get_prompt_text(self) -> str:
        """Generates plain text prompt descriptions of all tools."""
        return "\n".join([
            ToolSpecConverter.to_prompt_text(tool)
            for tool in self.tool_registry.values()
        ])

    async def run(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_args_override_map: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Retrieves the specified BaseTool, executes it via its run method, and returns a formatted Message.
        """
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return Message(
                role="tool",
                content=f"Error: Tool '{tool_name}' is not registered.",
                name=tool_name,
            )

        # Merge arguments with optional overrides
        execution_args = deepcopy(tool_args)
        if tool_args_override_map and tool_name in tool_args_override_map:
            execution_args.update(tool_args_override_map[tool_name])

        try:
            # Delegate execution to BaseTool.run()
            raw_output = await tool.run(**execution_args)

            if isinstance(raw_output, (dict, list)):
                content = json.dumps(raw_output, ensure_ascii=False)
            else:
                content = str(raw_output)

            return Message(role="tool", content=content, name=tool_name)
        except Exception as e:
            return Message(
                role="tool",
                content=f"Execution error on tool '{tool_name}': {str(e)}",
                name=tool_name,
            )