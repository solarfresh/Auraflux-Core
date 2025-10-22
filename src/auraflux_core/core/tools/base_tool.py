from abc import ABC, abstractmethod
from typing import Any, Dict

from auraflux_core.core.configs.logging_config import setup_logging
from auraflux_core.core.schemas.tools import ToolConfig


class BaseTool(ABC):
    """
    Abstract base class for all tools in the Twinrad system.

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