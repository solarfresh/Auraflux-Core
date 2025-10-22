from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by looking for .git directory or pyproject.toml.
    Walks up the directory tree until it finds a marker of the project root.
    """
    current_path = start_path.resolve()
    while current_path != current_path.parent:
        # Look for common project root markers
        if (current_path / ".git").is_dir() or (current_path / "pyproject.toml").is_file():
            return current_path
        current_path = current_path.parent
    # Fallback to the original path if no project root found
    return start_path.resolve()


def get_core_env_files() -> list[Path]:
    """
    Get list of environment files for core TwinRAD settings.
    Returns only existing files to avoid warnings.
    """
    root_env = find_project_root(Path(__file__)) / ".env"

    env_files = []
    if root_env.exists():
        env_files.append(root_env)

    return env_files

class TwinRADSettings(BaseSettings):
    """
    Core TwinRAD settings for the multi-agent red teaming framework.
    This class handles configuration for the core twinrad package only.
    Service-specific settings (server, dashboard) are handled separately.
    """
    model_config = SettingsConfigDict(
        env_file=get_core_env_files(),  # Dynamic env file discovery with validation
        env_file_encoding='utf-8',
        extra="ignore"
    )

    # Core TwinRAD settings
    log_level: str = "INFO"

    # LLM API Keys for red team agents
    gooelg_genai_api_key: str = ''
    twinkle_base_url: HttpUrl = HttpUrl('http://localhost')
    twinkle_api_key: str = ''

    # Red team specific settings
    max_rounds: int = 20
    conversation_timeout: int = 300  # seconds
    safety_threshold: float = 0.8

    # Tool API Keys
    google_search_engine_id: str = ''
    google_search_engine_api_key: str = ''
    google_search_engine_base_url: str = "https://customsearch.googleapis.com/customsearch/v1"

# Create an instance of the Settings class
settings = TwinRADSettings()