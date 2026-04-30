from typing import Any, Dict, List, Literal, Optional

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
    provider: str
    model: str
    system_message: str = Field(default='You are a helpful AI Assistant.')
    temperature: float = 0.7
    top_p: float = 0.95

    # Non-standard field required by our infrastructure for structured output.
    # Pydantic allows us to explicitly define this alongside standard params.
    output_schema: Optional[Dict[str, Any]] = Field(default=None)


class LLMResponse(BaseModel):
    """A standardized schema for all model inference responses."""
    text: str
    token_usage: int = 0


class ModelConfig(BaseModel):
    id: str
    name: str
    max_model_len: int = 8192
    restrict_user_assistant_alternate: bool = False


class ProviderConfig(BaseModel):
    """
    Defines the configuration for a single model or API.

    Attributes:
    provider_type (Literal): The mode of the model, e.g., "gemini", "generic_api", "openai", "vllm".
    base_url (Optional[str]): The base URL for the API, if applicable.
    api_key (Optional[str]): The API key for authentication, if applicable.
    restrict_user_assistant_alternate (bool): Whether to restrict messages to only user and assistant roles.
    tensor_parallel_size (int): The tensor parallel size for distributed models.
    device (str): The device to run the model on, e.g., "cpu", "cuda", "auto".
    dtype (str): The data type for model computations, e.g., "float32", "float16", "bfloat16", "auto".
    """
    id: str
    type: Literal["GOOGLE", "OPENAI", "VLLM"] = "GOOGLE"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    supported_families: List[ModelConfig] = []

    # tensor_parallel_size: int = 1
    # device: str = 'auto'
    # dtype: str = 'auto'


class ClientConfig(BaseModel):
    """The root configuration for the client module."""
    providers: List[ProviderConfig] = Field(default_factory=list)
    initialize_mode: Literal['create_task', 'run_forever'] = 'create_task'
    timeout_seconds: NonNegativeInt = 300
    sleep_interval_seconds: float = 0.1
