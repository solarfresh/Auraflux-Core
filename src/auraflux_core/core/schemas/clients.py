from typing import List, Literal, Optional

from pydantic import BaseModel, Field, NonNegativeInt

from auraflux_core.core.schemas.messages import Message


class LLMRequest(BaseModel):
    """
    A standardized schema for all model inference requests.

    Attributes:
    max_tokens (NonNegativeInt): The maximum number of tokens to generate.
    messages (List[Message]): A list of messages forming the conversation history.
    model (str): The name of the model to use for generation.
    system_message (str): The system message to set the behavior of the assistant.
    temperature (float): The sampling temperature.
    top_p (float): The nucleus sampling parameter.
    """
    max_tokens: NonNegativeInt = 4096
    messages: List[Message]
    model: str
    system_message: str = 'You are a helpful AI Assistant.'
    temperature: float = 0.7
    top_p: float = 0.95


class LLMResponse(BaseModel):
    """A standardized schema for all model inference responses."""
    text: str


class ModelConfig(BaseModel):
    """
    Defines the configuration for a single model or API.

    Attributes:
    name (str): The unique name of the model or API.
    mode (Literal): The mode of the model, e.g., "gemini", "generic_api", "openai", "vllm".
    base_url (Optional[str]): The base URL for the API, if applicable.
    api_key (Optional[str]): The API key for authentication, if applicable.
    restrict_user_assistant_alternate (bool): Whether to restrict messages to only user and assistant roles.
    tensor_parallel_size (int): The tensor parallel size for distributed models.
    device (str): The device to run the model on, e.g., "cpu", "cuda", "auto".
    dtype (str): The data type for model computations, e.g., "float32", "float16", "bfloat16", "auto".
    """
    name: str
    mode: Literal["gemini", "generic_api", "openai", "vllm"] = "generic_api"
    max_model_len: int = 8192
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    restrict_user_assistant_alternate: bool = False

    tensor_parallel_size: int = 1
    device: str = 'auto'
    dtype: str = 'auto'


class ClientConfig(BaseModel):
    """The root configuration for the client module."""
    models: List[ModelConfig] = Field(default_factory=list)
