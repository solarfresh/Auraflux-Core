import re

import pytest
from pydantic import ValidationError

from auraflux_core.core.schemas.messages import (OutputFormat, PromptConfig,
                                                 StageSpec)


class TestPromptConfig:
    """
    Test suite for PromptConfig, StageSpec, and DefaultAgentConfig models.
    Ensures correct serialization, language fallback logic, schema validation,
    and string placeholder consistency.
    """

    # =========================================================================
    # 1. Serialization & Deserialization Tests
    # =========================================================================
    def test_prompt_config_parsing_success(self):
        """
        Verify that a valid dictionary correctly parses into a PromptConfig instance
        with expected nested types and values.
        """
        raw_data = {
            "system": {
                "templates": {
                    "default": "System prompt",
                    "zh-TW": "系統提示"
                }
            },
            "stages": {
                "brainstorm": {
                    "templates": {"default": "Topic: {{topic}}"},
                    "template_variables": {"topic": "USER_INPUT"},
                    "output_format": "JSON",
                    "output_schema": {"type": "object"}
                }
            }
        }

        config = PromptConfig.model_validate(raw_data)

        assert config.system is not None
        assert config.system.get_template("zh-TW") == "系統提示"

        stage = config.get_stage_spec("brainstorm")
        assert stage is not None
        assert stage.output_format == OutputFormat.JSON
        assert stage.template_variables == {"topic": "USER_INPUT"}

    def test_prompt_config_parsing_invalid_enum(self):
        """
        Ensure Pydantic raises a PydanticValidationError when an invalid output_format
        enum value is provided in stage configuration.
        """
        raw_data = {
            "stages": {
                "test": {
                    "templates": {"default": "Hello"},
                    "output_format": "INVALID_FORMAT"  # Invalid enum value
                }
            }
        }

        with pytest.raises(ValidationError):
            PromptConfig.model_validate(raw_data)

    # =========================================================================
    # 2. Multi-language Template Fallback Tests
    # =========================================================================
    def test_template_language_fallback(self):
        """
        Verify the language fallback logic:
        1. Exact language match if present.
        2. Fallback to 'default' if specified language is missing.
        3. Return empty string if neither specified language nor 'default' exists.
        """
        spec = StageSpec(
            templates={
                "default": "Default template",
                "zh-TW": "繁體中文範本"
            }
        )

        # 1. Exact language match
        assert spec.get_template("zh-TW") == "繁體中文範本"

        # 2. Fallback to 'default' when language is missing
        assert spec.get_template("ja-JP") == "Default template"

        # 3. Return empty string when neither key nor 'default' exists
        empty_spec = StageSpec(templates={"zh-TW": "Only Traditional Chinese"})
        assert empty_spec.get_template("en-US") == ""

    # =========================================================================
    # 4. Template Variable Matching Tests
    # =========================================================================
    def test_template_variables_match_placeholders(self):
        """
        Statically verify that all {{ variable }} placeholders present in the prompt template
        have corresponding key definitions inside `template_variables`.
        """
        spec = StageSpec(
            templates={"default": "Hello {{name}}, welcome to {{city}}!"},
            template_variables={"name": "USER_NAME", "city": "USER_CITY"}
        )

        template_str = spec.get_template("default")

        # Extract {{ variable_name }} using regex
        extracted_vars = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", template_str))
        defined_vars = set(spec.template_variables.keys())

        # Assert placeholders strictly match defined keys
        assert extracted_vars == defined_vars

    # =========================================================================
    # 5. Optional Fields & Edge Cases Handling Tests
    # =========================================================================
    def test_prompt_config_optional_handling(self):
        """
        Ensure PromptConfig handles empty inputs and missing optional fields
        (e.g., system=None or stages=None) without raising unexpected runtime exceptions.
        """
        empty_config = PromptConfig()

        assert empty_config.system is None
        assert empty_config.stages == {} or empty_config.stages is None
        assert empty_config.get_stage_spec("non_existing_stage") is None
