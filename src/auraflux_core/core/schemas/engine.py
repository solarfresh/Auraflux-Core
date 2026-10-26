from pydantic import BaseModel, ConfigDict

from auraflux_core.core.schemas.clients import ClientConfig


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    client_config: ClientConfig
