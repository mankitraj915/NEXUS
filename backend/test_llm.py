import asyncio
import os
import time
from app.domains.search.query.llm_provider import OpenAICompatibleProvider

async def main():
    provider = OpenAICompatibleProvider()
    response = await provider.client.chat.completions.create(
        model=provider._model,
        messages=[
            {"role": "system", "content": "You are a search query transformation engine. Output strictly valid JSON with a single 'variants' array."},
            {"role": "user", "content": "What is the impact of BRCA1 mutation?"}
        ],
        temperature=provider.temperature,
        max_tokens=provider.max_tokens,
        response_format={"type": "json_object"}
    )
    print("Raw output:", repr(response.choices[0].message.content))

if __name__ == "__main__":
    asyncio.run(main())
