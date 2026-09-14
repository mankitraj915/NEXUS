# NEXUS Configuration Guide

NEXUS uses a centralized configuration system powered by `pydantic-settings`. All configuration options, including database credentials and LLM provider secrets, are loaded through the `app.config.settings.Settings` class.

## Configuration Precedence

When starting NEXUS or running benchmarks, configuration values are resolved in the following priority order (highest to lowest):

1. **Runtime / Programmatic configuration**: Explicitly overriding values in the code.
2. **Environment variables**: Variables explicitly exported in the shell session.
3. **`.env` file**: A local file containing key-value pairs at the root of the backend directory.
4. **Safe Defaults**: Non-secret default values defined in `app/config/settings.py`.

## Configuring Credentials for Phase 8B Benchmarks

For the LLM multi-query generation (Phase 8B), NEXUS requires access to an OpenAI-compatible API.

**Never hardcode credentials or place them in source code.** 
Ensure your `.env` file is never committed to Git (it is already included in `.gitignore`).

### Option A: Local `.env` File (Recommended for Local Dev)

Copy the `.env.example` file to `.env`:

```bash
copy .env.example .env
```

Edit the `.env` file to include your API key:
```text
OPENAI_API_KEY=your-secret-key-here
OPENAI_MODEL=gpt-4o-mini
```

Run the benchmark using `uv run`, which will automatically load the `.env` file:
```powershell
uv run python -m scripts.benchmark generate-llm-variants --dataset scifact
```

### Option B: PowerShell Session Variable (Recommended for CI / Secure Environments)

You can set the credential directly in your active PowerShell session. This explicitly overrides any value in the `.env` file.

```powershell
$env:OPENAI_API_KEY="your-secret-key-here"
```

Then execute the benchmark:
```powershell
uv run python -m scripts.benchmark generate-llm-variants --dataset scifact
```

When you close the PowerShell session, the key will be securely discarded from memory.

## Pre-flight Checks

Before running the full LLM multi-query generation (which can take a while and cost API credits), you can verify your configuration and credentials.

### Configuration Preflight
Validates that credentials exist, are correctly formatted, and that the cache is ready. It **does not** make any external API requests.
```powershell
uv run python -m scripts.benchmark generate-llm-variants --dataset scifact --preflight
```

### Smoke Test
Validates the full pipeline by generating variants for exactly **one** query and verifying semantic drift filtering.
```powershell
uv run python -m scripts.benchmark generate-llm-variants --dataset scifact --smoke-test
```
