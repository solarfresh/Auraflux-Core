from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field


class ToolCallProtocol(Enum):
    """
    Defines the communication protocol used to trigger a tool call between the Agent and the LLM.
    """

    PROMPT = "PROMPT"
    """
    Manual JSON injection mode.
    The tool definitions are appended to the system message, and the LLM is instructed
    to output a structured string (e.g., Markdown JSON blocks) which is then parsed
    locally using regular expressions or JSON decoders.
    """

    NATIVE = "NATIVE"
    """
    Model-native function calling mode.
    Uses the provider's official API parameters (e.g., OpenAI's 'tools' or Gemini's
    'function_declarations'). This leverages model-specific fine-tuning and
    constrained decoding for higher reliability and structured finish reasons.
    """


class ToolExecutionStrategy(Enum):
    """
    Defines how the orchestrator handles the lifecycle and output of a tool execution
    once the tool parameters have been determined.
    """

    NONE = "NONE"
    """
    Standard text generation mode.
    No tools are involved in this turn.
    """

    DIRECT = "DIRECT"
    """
    Router or 'One-Shot' execution mode.
    The raw output of the tool is returned directly as the final message.
    The LLM acts as a router, and no further synthesis or natural language
    processing is performed on the tool result.
    """

    REFLECTIVE = "REFLECTIVE"
    """
    Reasoning or 'And-Process' mode.
    The tool output is appended to the message history and sent back to the LLM.
    The model then synthesizes, explains, or reasons based on the tool results
    to provide a final conversational answer.
    """


class ToolConfig(BaseModel):
    pass

