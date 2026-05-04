from abc import ABC, abstractmethod
from typing import Any, Dict

from google.genai import types

from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.tools import ToolConfig


class BaseTool(ABC):
    """
    Abstract base class for all tools in the Auraflux system.

    This class defines the required interface for any tool to be
    integrated with an agent.
    """

    def __init__(self, config: ToolConfig = ToolConfig()) -> None:
        self.config = config
        self.logger = setup_logging(name=f"[{self.__class__.__name__}]")

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """
        An abstract method that must be overridden by all subclasses.
        It should contain the core logic of the tool.

        Args:
            **kwargs: Arbitrary keyword arguments representing the tool's input parameters.

        Returns:
            Any: The result of the tool's operation.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the tool.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Returns a brief description of the tool's function.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns a dictionary of the tool's parameters, including their type and description.
        This is crucial for LLMs to understand how to call the tool.
        """
        pass


class ToolSpecConverter:

    @staticmethod
    def to_openai(tool: BaseTool) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "parameters": tool.get_parameters(),
            }
        }

    @staticmethod
    def to_gemini(tool: BaseTool) -> Any:
        raw_params = tool.get_parameters()

        return types.FunctionDeclaration(
            name=tool.get_name(),
            description=tool.get_description(),
            parameters=ToolSpecConverter._dict_to_gemini_schema(raw_params)
        )

    @staticmethod
    def to_prompt_text(tool: BaseTool) -> str:
        return f"Tool: {tool.get_name()}\nDescription: {tool.get_description()}\nArgs: {tool.get_parameters()}\n"

    @staticmethod
    def _dict_to_gemini_schema(schema_dict: Dict[str, Any]) -> types.Schema:
        properties = None
        if "properties" in schema_dict:
            properties = {
                k: ToolSpecConverter._dict_to_gemini_schema(v)
                for k, v in schema_dict["properties"].items()
            }

        return types.Schema(
            type=schema_dict.get("type", "object").upper(), # Gemini 要求大寫 (如 OBJECT, STRING)
            description=schema_dict.get("description"),
            properties=properties,
            required=schema_dict.get("required"),
            items=ToolSpecConverter._dict_to_gemini_schema(schema_dict["items"]) if "items" in schema_dict else None,
            enum=schema_dict.get("enum"),
            format=schema_dict.get("format")
        )
