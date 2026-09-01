from typing import Any, Dict, NamedTuple, Type

from auraflux_core.core.embeddings.generic_embedding import GenericEmbedding
from auraflux_core.core.schemas.embeddings import EmbeddingConfig


class EmbeddingImplementation(NamedTuple):
    embedding_class: Type[Any]
    config_class: Type[Any]


EMBEDDING_REGISTRY: Dict[str, EmbeddingImplementation] = {
    'default': EmbeddingImplementation(
        embedding_class=GenericEmbedding,
        config_class=EmbeddingConfig
    )
}