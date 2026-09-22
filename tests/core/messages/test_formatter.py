from auraflux_core.core.messages.formatter import PromptFormatter
from auraflux_core.core.schemas.messages import (Message, PromptConfig,
                                                 StageSpec, SystemPromptSpec)


class TestPromptFormatter:
    """Unit test suite for validating system message resolution and stage message formatting."""

    def test_format_system_message_success(self, mock_agent_config_factory):
        """Verify successful resolution and formatting of system message from PromptConfig."""
        config = mock_agent_config_factory()
        config.prompt_config = PromptConfig(
            system=SystemPromptSpec(templates={"default": "System Prompt: {role}"})
        )
        formatter = PromptFormatter(config=config)

        result = formatter.format_system_message(role="Assistant")
        assert result == "System Prompt: Assistant"

    def test_format_system_message_missing_config(self, mock_agent_config_factory):
        """Verify safe empty string return when prompt_config or system spec is absent."""
        config = mock_agent_config_factory()
        config.prompt_config = None
        formatter = PromptFormatter(config=config)

        assert formatter.format_system_message() == ""

    def test_format_stage_message_success(self, mock_agent_config_factory):
        """Verify successful user message formatting with dynamic template parameters."""
        config = mock_agent_config_factory(lang="zh-TW")
        config.prompt_config = PromptConfig(
            stages={
                "planning": StageSpec(
                    templates={
                        "default": "Analyze text: {text}",
                        "zh-TW": "分析文本：{text}",
                    }
                )
            }
        )
        formatter = PromptFormatter(config=config)

        message = formatter.format_stage_message(
            agent_name="keyword_extractor",
            stage_name="planning",
            text="Hello Agent",
        )
        assert isinstance(message, Message)
        assert message.role == "user"
        assert message.content == "分析文本：Hello Agent"

    def test_format_stage_message_language_fallback(self, mock_agent_config_factory):
        """Verify fallback to 'default' template when requested language code is unavailable."""
        config = mock_agent_config_factory(lang="ja-JP")
        config.prompt_config = PromptConfig(
            stages={
                "planning": StageSpec(
                    templates={"default": "Analyze text: {text}"}
                )
            }
        )
        formatter = PromptFormatter(config=config)

        message = formatter.format_stage_message(
            agent_name="keyword_extractor",
            stage_name="planning",
            text="Test Fallback",
        )
        assert message.content == "Analyze text: Test Fallback"

    def test_format_stage_message_non_existent_stage(self, mock_agent_config):
        """Verify safe handling and empty user message output when querying an unconfigured stage."""
        formatter = PromptFormatter(config=mock_agent_config)

        message = formatter.format_stage_message(
            agent_name="keyword_extractor",
            stage_name="non_existent_stage",
        )
        assert isinstance(message, Message)
        assert message.role == "user"
        assert message.content == ""
