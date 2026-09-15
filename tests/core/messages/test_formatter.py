from auraflux_core.core.schemas.messages import Message, PromptConfig, PromptSpec
from auraflux_core.core.messages.formatter import PromptFormatter


class TestPromptFormatter:
    """Unit test suite for validating system message resolution and stage message formatting."""

    def test_format_system_message_precedence(self, mock_agent_config_factory):
        """Verify precedence order for system message resolution:

        1. PromptConfig.system
        2. AgentConfig.system_message
        3. Legacy system_message_map
        """
        # Scenario A: PromptConfig.system takes highest precedence
        config_a = mock_agent_config_factory(system_message="Config Direct System")
        prompt_config_a = PromptConfig(
            system=PromptSpec(templates={"default": "PromptConfig System"})
        )
        formatter_a = PromptFormatter(config=config_a, prompt_config=prompt_config_a)
        assert formatter_a.format_system_message() == "PromptConfig System"

        # Scenario B: Fall back to config.system_message when PromptConfig.system is missing
        config_b = mock_agent_config_factory(system_message="Config Direct System")
        formatter_b = PromptFormatter(config=config_b)
        assert formatter_b.format_system_message() == "Config Direct System"

        # Scenario C: Fall back to legacy system_message_map
        config_c = mock_agent_config_factory()
        formatter_c = PromptFormatter(
            config=config_c, system_message_map={"default": "Legacy Map System"}
        )
        assert formatter_c.format_system_message() == "Legacy Map System"

    def test_format_stage_message_success(self, mock_agent_config_factory):
        """Verify successful user message formatting with dynamic template parameters."""
        config = mock_agent_config_factory(lang="zh-TW")
        prompt_config = PromptConfig(
            stages={
                "planning": PromptSpec(
                    templates={
                        "default": "Analyze text: {text}",
                        "zh-TW": "分析文本：{text}",
                    }
                )
            }
        )
        formatter = PromptFormatter(config=config, prompt_config=prompt_config)

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
        prompt_config = PromptConfig(
            stages={
                "planning": PromptSpec(
                    templates={"default": "Analyze text: {text}"}
                )
            }
        )
        formatter = PromptFormatter(config=config, prompt_config=prompt_config)

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
