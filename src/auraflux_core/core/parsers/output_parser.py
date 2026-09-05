import json
import re
from typing import Any, Dict


class OutputParser:
    """
    Handles parsing and cleaning of LLM outputs, including Markdown code block extraction,
    LaTeX symbol sanitization, CoT tag stripping, and JSON parsing.
    """

    def parse_json(self, output_string: str) -> Dict[str, Any]:
        """
        Extracts and parses JSON object from an raw LLM response string.

        Args:
            output_string (str): Raw response text from LLM.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary.

        Raises:
            ValueError: If parsing fails after cleaning.
        """
        # Step 1: Strip thinking process (<think>...</think>) if present
        clean_text = self.strip_thinking_tags(output_string)

        # Step 2: Try extracting from Markdown JSON block (```json ... ```)
        json_pattern = r"```json\s*(\{.*\}|\[.*\])\s*```"
        match = re.search(json_pattern, clean_text, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            json_str = clean_text.strip()

        # Step 3: Sanitize potential broken characters (e.g., LaTeX formulas)
        json_str = self.sanitize_json_string(json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON output: {e}\nRaw Output: {output_string}") from e

    def strip_thinking_tags(self, text: str) -> str:
        """
        Removes Reasoning/CoT <think>...</think> blocks from response text.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def sanitize_json_string(self, json_string: str) -> str:
        """
        Sanitizes raw string before JSON loading (e.g. cleaning unescaped LaTeX symbols).
        """
        clean_string = re.sub(r'\\\w+\{([^}]+)\}', r'->(\1)->', json_string)
        clean_string = clean_string.replace('$', '')
        return clean_string
