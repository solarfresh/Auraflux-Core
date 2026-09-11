from typing import Any, Dict, Optional

from auraflux_core.core.agents.pipelines.plan_and_execute import \
    PlanAndExecuteHandler


class AlignmentHandler(PlanAndExecuteHandler):
    """Abstract Base Handler providing shared retrieval tool specs and helper utilities."""

    TARGET_FIELD_MAP = {
        "question": {"text": "target_question", "vector": "question_vector"},
        "concept_title": {"text": "concept_title", "vector": "concept_vector"},
        "concept_desc": {"text": "concept_description", "vector": "concept_vector"},
        "evidence": {"text": "evidence_text", "vector": "evidence_vector"}
    }

    def _clean_field_name(self, raw_field: Any, fallback: str) -> str:
        """Strips List structures or Boost annotations (e.g. ['evidence_text^2.0'] -> 'evidence_text')."""
        if isinstance(raw_field, list) and raw_field:
            field = str(raw_field[0])
        elif isinstance(raw_field, str) and raw_field:
            field = raw_field
        else:
            field = fallback
        return field.split("^")[0].strip()

    def _build_retriever_spec(self, payload: Dict[str, Any], plan_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Shared logic for extracting hybrid_retriever tool specifications."""
        raw_queries = plan_output.get("queries", [])
        triples = plan_output.get("triples", [])

        # Fallback to triples if queries are empty
        if not isinstance(raw_queries, list) or not raw_queries:
            if isinstance(triples, list) and triples:
                t = triples[0]
                constructed_query = f"{t.get('subject', '')} {t.get('object', '')}".strip()
                if constructed_query:
                    raw_queries = [{
                        "query_text": constructed_query,
                        "target_type": "evidence",
                        "text_field": "evidence_text",
                        "vector_field": "evidence_vector"
                    }]

        if not raw_queries:
            return None

        query_items = []
        for q in raw_queries:
            if not isinstance(q, dict) or not q.get("query_text"):
                continue
            target_type = q.get("target_type", "evidence")
            target_config = self.TARGET_FIELD_MAP.get(target_type) or self.TARGET_FIELD_MAP.get("evidence", {})

            t_field = self._clean_field_name(q.get("text_field"), str(target_config.get("text")))
            v_field = self._clean_field_name(q.get("vector_field"), str(target_config.get("vector")))

            query_items.append({
                "query_text": q["query_text"],
                "text_field": t_field,
                "vector_field": v_field
            })

        if not query_items:
            return None

        tool_args = {"query_items": query_items, "top_k": plan_output.get("top_k", 5)}
        if payload.get("routing_key"):
            tool_args["routing"] = str(payload["routing_key"])
        if payload.get("index_name"):
            tool_args["index_name"] = str(payload["index_name"])

        return {"tool_name": "hybrid_retriever", "tool_args": tool_args}
