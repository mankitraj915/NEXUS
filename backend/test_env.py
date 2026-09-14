from app.config.settings import settings
print(f"API KEY from settings: {settings.model_extra.get('OPENAI_API_KEY')}")
