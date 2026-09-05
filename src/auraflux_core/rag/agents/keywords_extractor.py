import json
from typing import Dict, List

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.rag.config.regex_patterns import KeywordPatternCollection, DEFAULT_KEYWORD_PATTERNS


class ExtractKeywordsAgent(BaseAgent):

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "default": (
                "You are an expert Information Extraction Agent specializing in Knowledge Graph construction and Document Analysis.\n"
                "Your task is to analyze the provided raw document chunk text and extract structured key information: Bound Semantic Triples and Exact Keywords/Entities.\n\n"
                "### EXTRACTION RULES\n\n"
                "1. **Semantic Triples (`triples`)**:\n"
                "   - Extract bound facts expressed as closed-world triples: `(subject, predicate, object)`.\n"
                "   - **EXACT TEXT MATCH REQUIRED**: `subject`, `predicate`, and `object` MUST strictly use the exact phrasing, exact terms, and wording present in the source text. Do NOT paraphrase, summarize, normalize, or translate any of the words into synonyms.\n"
                "   - `subject`: Core entity, concrete object, actor, or specific parameter name directly quoted from text. **NO PRONOUNS**: Do NOT use vague pronouns or general self-references (e.g., 'we', 'I', 'they', 'it', '我們', '這個') as the subject. If a subject lacks explicit technical meaning, skip the triple.\n"
                "   - `predicate`: The action, relation, logical condition, or operator connecting subject and object as verbatim from text.\n"
                "   - `object`: Target entity, metric, quantitative limit, or constrained value directly quoted from text.\n"
                "   - **FACTUAL RELATIONS ONLY**: Extract only explicit, concrete, operational, or logical relationships. Do NOT extract poetic metaphors, analogies, or rhetorical comparisons (e.g., skip statements like 'X is like Y learning to drive').\n"
                "   - **Length Limit**: Keep `subject`, `predicate`, and `object` concise (under 10 words each). Do NOT insert entire sentences into a triple field.\n\n"
                "2. **Exact Keywords & Entities (`tags`)**:\n"
                "   - Extract ALL distinct key terms, proper nouns, domain concepts, and technical entities directly quoted from the source text.\n"
                "   - **NO QUANTITY LIMIT**: Extract every valid entity present; do not cap or artificially limit the count.\n"
                "   - **EXACT TEXT MATCH REQUIRED**: Every item MUST be a verbatim string copied directly from the text (preserving original casing, spelling, and hyphenation).\n\n"
                "   **Eligible Entity Categories**:\n"
                "   - **Proper Nouns & Entities**: Names of people, organizations, standards, locations, or products (e.g., \"John Doe\", \"Acme Corp\", \"ISO 27001\", \"Project Apollo\").\n"
                "   - **Domains & Frameworks**: Field titles, academic theories, or overall domain names (e.g., \"Cybersecurity\", \"Zero Trust Architecture\", \"Agile Methodology\").\n"
                "   - **Technical Terms & Methodologies**: Specific algorithms, architecture components, or technical mechanisms (e.g., \"AES-256\", \"Load Balancer\", \"Gradient Descent\").\n"
                "   - **Key Operational Concepts**: Core terms, parameters, or policies explicitly discussed in context (e.g., \"Retention Period\", \"Authentication Factor\", \"Throughput Limit\").\n\n"
                "   **Exclusion Rules**:\n"
                "   - Exclude generic common nouns lacking unique semantic context (e.g., \"document\", \"item\", \"example\", \"result\").\n"
                "   - Do NOT split multi-word technical concepts or titles into separated atomic words (e.g., extract \"Zero Trust Architecture\", NOT just \"Architecture\").\n\n"
                "3. **Empty / Low-Information Chunks**:\n"
                "   - If the input text consists solely of headers, layout metadata, conversational transitions, or lacks substantive facts, return empty structures for both triples and tags.\n"
                "   - Do NOT force extraction if no explicit entities or domain keywords exist in the text.\n\n"
                "### OUTPUT STRUCTURE SPECIFICATION\n\n"
                "The output contains two primary fields:\n\n"
                "* **`triples`**: A list of structured items capturing precise semantic relationships using verbatim text.\n"
                "  - `subject`: (String) Exact source entity or subject appearing in the text.\n"
                "  - `predicate`: (String) Exact linking operator or relationship phrase appearing in the text.\n"
                "  - `object`: (String) Exact target entity or constraint value appearing in the text.\n"
                "* **`tags`**: (List of Strings) An unconstrained list of exact key terms, proper nouns, domain concepts, and technical entities extracted directly from the text as verbatim strings.\n"
            )
        }

    def postprocess_llm_output(self, output_string: str) -> str:
        """Processes and cleans the semantic data structure returned by the Agent."""
        json_object = self.output_parser.parse_json(output_string)

        # 1. Clean tags
        if "tags" in json_object and isinstance(json_object["tags"], list):
            json_object["tags"] = [
                cleaned
                for tag in json_object["tags"]
                if (cleaned := self.clean_extracted_text(tag))
            ]

        # 2. Clean triples (subject, predicate, object)
        if "triples" in json_object and isinstance(json_object["triples"], list):
            json_object["triples"] = self.process_triples(json_object["triples"])

        return json.dumps(json_object, ensure_ascii=False)

    def clean_extracted_text(self, text: str) -> str:
        """Removes leading and trailing invalid punctuation marks from a single string."""
        # Define quote brackets and punctuation to be stripped (includes Chinese title brackets, standard quotes, etc.)
        STRIP_CHARS = '《》""''「」『』【】（）()〈〉'
        if not text:
            return text
        # Strip whitespace and specified punctuation marks from both ends
        cleaned = text.strip().strip(STRIP_CHARS).strip()
        return cleaned

    def process_triples(
        self,
        triples: List[Dict[str, str]],
        patterns: KeywordPatternCollection = DEFAULT_KEYWORD_PATTERNS
    ) -> List[Dict[str, str]]:
        """
        Cleans triples and splits enumerated objects into individual atomic triples.
        Avoids duplicate function calls during iteration by utilizing the walrus operator.

        Args:
            triples: List of raw triple dictionaries containing subject, predicate, and object.
            patterns: Pattern collection containing the enumeration splitting regex.

        Returns:
            List of cleaned, single-object atomic triples.
        """
        processed_triples: List[Dict[str, str]] = []

        for item in triples:
            subj = self.clean_extracted_text(item.get("subject", ""))
            pred = self.clean_extracted_text(item.get("predicate", ""))
            obj_raw = self.clean_extracted_text(item.get("object", ""))

            if not (subj and pred and obj_raw):
                continue

            # Split enumerated objects and clean each item only once using the walrus operator
            split_objects = [
                cleaned
                for o in patterns.enumeration_splitter.split(obj_raw)
                if (cleaned := self.clean_extracted_text(o))
            ]

            # Expand into individual atomic triples
            for single_obj in split_objects:
                processed_triples.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": single_obj
                })

        return processed_triples
