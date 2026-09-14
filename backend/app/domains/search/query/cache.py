import hashlib
import json
import os
import sqlite3
from typing import Any


class VariantCache:
    def __init__(self, db_path: str = ".cache/llm_variants.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS variant_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT,
                    variants_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def generate_key(self, query: str, config: dict[str, Any]) -> str:
        """
        config should include: transformer_version, provider, model,
        model_version, prompt_version, generation_parameters, max_variants, drift_threshold
        """
        payload = {"query": query, "config": config}
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT variants_json FROM variant_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set(self, cache_key: str, query: str, variants_data: dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO variant_cache (cache_key, query, variants_json) VALUES (?, ?, ?)",
                (cache_key, query, json.dumps(variants_data)),
            )
            conn.commit()
