from typing import Any, Dict, List, Optional

from auraflux_core.core.schemas.tools import ToolConfig
from auraflux_core.core.tools.base_tool import BaseTool
from auraflux_core.rag.config.regex_patterns import (DEFAULT_KEYWORD_PATTERNS,
                                                     KeywordPatternCollection)


class TripleProcessorTool(BaseTool):
    """
    Tool for cleaning, object-splitting, and metric normalizing structured semantic triples.
    """

    def __init__(
        self,
        patterns: KeywordPatternCollection = DEFAULT_KEYWORD_PATTERNS,
        config: ToolConfig = ToolConfig()
    ) -> None:
        super().__init__(config=config)
        self.patterns = patterns

    async def run(
        self,
        triples: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Executes triple cleaning, enumeration splitting, and metric formatting.
        Implements BaseTool.run() abstract method.
        """
        self.logger.info(f"Processing {len(triples)} semantic triples.")
        return self.process_triples(triples, patterns=self.patterns)

    def process_triples(
        self,
        triples: List[Dict[str, Any]],
        patterns: KeywordPatternCollection = DEFAULT_KEYWORD_PATTERNS
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates triple processing: cleaning, object splitting, and metric formatting.
        """
        processed_triples: List[Dict[str, Any]] = []

        for item in triples:
            subj = self.clean_extracted_text(item.get("subject", ""))
            pred = self.clean_extracted_text(item.get("predicate", ""))
            obj_raw = self.clean_extracted_text(item.get("object", ""))

            if not (subj and pred and obj_raw):
                continue

            norm_val = self._normalize_metric_value(item.get("normalized_value"))
            split_objects = self._split_enumerated_object(obj_raw, patterns)

            for single_obj in split_objects:
                triple_dict = self._build_triple_dict(
                    subj=subj,
                    pred=pred,
                    obj=single_obj,
                    item=item,
                    norm_val=norm_val
                )
                processed_triples.append(triple_dict)

        return processed_triples

    def clean_extracted_text(self, text: str) -> str:
        """Removes leading and trailing invalid punctuation marks from a single string."""
        STRIP_CHARS = '《》""''「」『』【】（）()〈〉'
        if not text:
            return text
        return text.strip().strip(STRIP_CHARS).strip()

    def _split_enumerated_object(
        self,
        obj_raw: str,
        patterns: KeywordPatternCollection
    ) -> List[str]:
        primary_delimiter_split_chunks = patterns.explicit_delimiter_splitter.split(obj_raw)

        primary_split_objects = [
            cleaned
            for o in primary_delimiter_split_chunks
            if (cleaned := self.clean_extracted_text(o))
        ]

        final_split_objects: List[str] = []
        is_multi_item_enumeration = len(primary_split_objects) > 1

        for chunk in primary_split_objects:
            cleaned_chunk = self.clean_extracted_text(chunk)

            if is_multi_item_enumeration and patterns.contextual_enumeration_splitter.search(cleaned_chunk):
                secondary_chunks = patterns.contextual_enumeration_splitter.split(cleaned_chunk)
                final_split_objects.extend([
                    cleaned
                    for sub in secondary_chunks
                    if (cleaned := self.clean_extracted_text(sub))
                ])
            else:
                final_split_objects.append(cleaned_chunk)

        return final_split_objects

    def _normalize_metric_value(self, raw_val: Any) -> Optional[float]:
        if raw_val is None:
            return None
        try:
            return float(raw_val)
        except (ValueError, TypeError):
            return None

    def _build_triple_dict(
        self,
        subj: str,
        pred: str,
        obj: str,
        item: Dict[str, Any],
        norm_val: Optional[float]
    ) -> Dict[str, Any]:
        triple_dict: Dict[str, Any] = {
            "subject": subj,
            "predicate": pred,
            "object": obj
        }

        metric_name = item.get("metric_name")
        unit = item.get("unit")
        operator = item.get("operator", "<=")

        if metric_name and norm_val is not None:
            triple_dict.update({
                "metric_name": str(metric_name),
                "normalized_value": norm_val,
                "unit": str(unit) if unit else "",
                "operator": str(operator)
            })

        return triple_dict

    def get_name(self) -> str:
        """Returns the unique tool identifier for LLM tool invocation."""
        return "triple_processor"

    def get_description(self) -> str:
        """Returns the function description used by LLMs to determine tool routing."""
        return (
            "Cleans, splits enumerated objects, and formats metric metadata "
            "for extracted knowledge graph semantic triples."
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Generates JSON Schema parameter specs for LLM function calling."""
        return {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "description": "A list of raw triple dictionaries containing subject, predicate, and object.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "metric_name": {"type": "string"},
                            "normalized_value": {"type": "number"},
                            "unit": {"type": "string"},
                            "operator": {"type": "string"}
                        },
                        "required": ["subject", "predicate", "object"]
                    }
                }
            },
            "required": ["triples"]
        }
