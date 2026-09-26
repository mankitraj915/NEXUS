import pytest

from app.config.settings import Settings
from app.domains.search.query.llm_provider import OpenAICompatibleProvider


@pytest.fixture
def clear_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


def test_missing_key_fails(clear_env):
    # Test 1: No key -> missing configuration
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        # We need to monkeypatch the global settings used by validate_configuration
        import app.domains.search.query.llm_provider as provider_module

        provider_module.settings = settings
        OpenAICompatibleProvider.validate_configuration()


def test_environment_key_works(clear_env, monkeypatch):
    # Test 2: Environment key -> works
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    import app.domains.search.query.llm_provider as provider_module

    provider_module.settings = settings

    # Should not raise
    OpenAICompatibleProvider.validate_configuration()

    provider = OpenAICompatibleProvider()
    assert provider.client.api_key == "env-key"


def test_env_file_works(clear_env, tmp_path):
    # Test 3: .env key -> works
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n")

    settings = Settings(_env_file=str(env_file))  # type: ignore[call-arg]

    import app.domains.search.query.llm_provider as provider_module

    provider_module.settings = settings

    OpenAICompatibleProvider.validate_configuration()
    provider = OpenAICompatibleProvider()
    assert provider.client.api_key == "file-key"


def test_environment_overrides_env_file(clear_env, tmp_path, monkeypatch):
    # Test 4: Environment overrides .env
    monkeypatch.setenv("OPENAI_API_KEY", "env-override-key")

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n")

    settings = Settings(_env_file=str(env_file))  # type: ignore[call-arg]

    import app.domains.search.query.llm_provider as provider_module

    provider_module.settings = settings

    OpenAICompatibleProvider.validate_configuration()
    provider = OpenAICompatibleProvider()
    assert provider.client.api_key == "env-override-key"


def test_whitespace_key_rejected(clear_env):
    # Test 5: Whitespace-only key fails validation
    settings = Settings(OPENAI_API_KEY="   ", _env_file=None)  # type: ignore[call-arg]

    import app.domains.search.query.llm_provider as provider_module

    provider_module.settings = settings

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        OpenAICompatibleProvider.validate_configuration()


def test_provider_does_not_log_secrets(clear_env, caplog):
    # Test 6: Provider initialization does not log secrets
    settings = Settings(OPENAI_API_KEY="super-secret-key", _env_file=None)  # type: ignore[call-arg]

    import app.domains.search.query.llm_provider as provider_module

    provider_module.settings = settings

    provider = OpenAICompatibleProvider()

    # Log string representation to ensure it doesn't leak secrets via __str__
    import logging

    logger = logging.getLogger("test")
    logger.info(f"Provider: {provider}")

    assert "super-secret-key" not in caplog.text
