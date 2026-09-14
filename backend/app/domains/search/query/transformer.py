from abc import ABC, abstractmethod
from typing import List
from datetime import datetime, timezone
import hashlib
import json
import os
from app.domains.search.query.models import QueryVariant
from app.domains.search.query.features import STOP_WORDS, CONJUNCTIONS
from app.domains.search.query.llm_provider import LLMProvider
from app.domains.search.query.drift import SemanticDriftFilter
from app.domains.search.query.cache import VariantCache

class QueryTransformer(ABC):
    @abstractmethod
    async def transform(self, query: str) -> List[QueryVariant]:
        pass

class RuleBasedQueryTransformer(QueryTransformer):
    """
    Deterministic rule-based transformer.
    Generates:
    - Original Query (always)
    - Lexical Expansion (keyword version by removing stopwords)
    - Question Decomposition (if conjunctives like "and" are found)
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.generator = "rule_based"
        
    def _create_variant(self, query: str, vtype: str, is_original: bool = False, v_id: str = None) -> QueryVariant:
        if v_id is None:
            v_id = hashlib.md5((query + vtype + self.version).encode()).hexdigest()[:8]
        return QueryVariant(
            variant_id=v_id,
            query=query,
            variant_type=vtype,
            generator=self.generator,
            generator_version=self.version,
            confidence=1.0,
            is_original=is_original,
            created_at=datetime.now(timezone.utc)
        )
        
    async def transform(self, query: str) -> List[QueryVariant]:
        variants = []
        
        # 1. Original Query
        variants.append(self._create_variant(query, "original", is_original=True, v_id="Q0"))
        
        # 2. Lexical Expansion (Keyword focus)
        tokens = query.split()
        keyword_tokens = [t for t in tokens if t.lower() not in STOP_WORDS]
        keyword_query = " ".join(keyword_tokens)
        if keyword_query and keyword_query != query:
            variants.append(self._create_variant(keyword_query, "lexical_expansion"))
            
        # 3. Question Decomposition (Split on conjunctives)
        lower_tokens = [t.lower() for t in tokens]
        for conj in CONJUNCTIONS:
            if conj in lower_tokens:
                idx = lower_tokens.index(conj)
                if 0 < idx < len(tokens) - 1:
                    part1 = " ".join(tokens[:idx])
                    part2 = " ".join(tokens[idx+1:])
                    # To avoid tiny fragments, only add if they are at least 2 tokens
                    if len(part1.split()) >= 2:
                        variants.append(self._create_variant(part1, "question_decomposition"))
                    if len(part2.split()) >= 2:
                        variants.append(self._create_variant(part2, "question_decomposition"))
                    break # Just split once for simplicity
                    
        return variants

class TransformationCache:
    def __init__(self, cache_file=".cache/transformations.json"):
        self.cache_file = cache_file
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                pass
                
    def get(self, query: str, generator_version: str) -> List[QueryVariant]:
        key = hashlib.md5((query + generator_version).encode()).hexdigest()
        if key in self.cache:
            data = self.cache[key]
            return [QueryVariant(**v) for v in data]
        return None
        
    def set(self, query: str, generator_version: str, variants: List[QueryVariant]):
        key = hashlib.md5((query + generator_version).encode()).hexdigest()
        self.cache[key] = [v.model_dump() for v in variants]
        # Very simple flush to disk
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w") as f:
            # Pydantic v2 datetimes need custom serialization in json.dump, 
            # but model_dump(mode='json') handles it. Let's do that.
            self.cache[key] = [v.model_dump(mode='json') for v in variants]
            json.dump(self.cache, f)

class LLMQueryTransformer(QueryTransformer):
    def __init__(self, provider: LLMProvider, drift_filter: SemanticDriftFilter, cache: VariantCache, max_variants: int = 3):
        self.provider = provider
        self.drift_filter = drift_filter
        self.cache = cache
        self.max_variants = max_variants
        self.version = "phase8b-v1"
        self.generator = "llm_" + provider.provider_name

    def _get_config_dict(self):
        return {
            "transformer_version": self.version,
            "provider": self.provider.provider_name,
            "model": self.provider.model_name,
            "max_variants": self.max_variants,
            "drift_threshold": self.drift_filter.threshold
        }

    async def transform(self, query: str) -> List[QueryVariant]:
        # Always create the original query variant
        original_variant = QueryVariant(
            variant_id="Q0",
            query=query,
            variant_type="original",
            generator=self.generator,
            generator_version=self.version,
            confidence=1.0,
            is_original=True,
            created_at=datetime.now(timezone.utc)
        )
        
        config = self._get_config_dict()
        cache_key = self.cache.generate_key(query, config)
        
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            variants = [original_variant]
            for v_data in cached_data.get("variants", []):
                if v_data.get("is_accepted"):
                    v_id = hashlib.md5((query + v_data["query"] + self.version).encode()).hexdigest()[:8]
                    variants.append(QueryVariant(
                        variant_id=v_id,
                        query=v_data["query"],
                        variant_type=v_data.get("type", "UNKNOWN"),
                        generator=self.generator,
                        generator_version=self.version,
                        confidence=v_data.get("confidence", 0.9),
                        is_original=False,
                        created_at=datetime.now(timezone.utc)
                    ))
            # Slice to max_variants (excluding original)
            return variants[:self.max_variants + 1]

        prompt = f"""You are a search query transformation engine.
Your task is to generate up to {self.max_variants} alternative search query formulations for the given original query.

RULES:
1. Preserve the original information need exactly.
2. Produce retrieval-oriented reformulations (e.g., keyword extraction, technical synonyms).
3. Do NOT answer the question.
4. Do NOT add unsupported entities or hallucinate constraints.
5. Do NOT change polarity.

Output strictly valid JSON with a single 'variants' array. Each item should have:
- "query": The transformed query string
- "type": One of [TECHNICAL_REFORMULATION, KEYWORD_REFORMULATION, ENTITY_FOCUSED, CONCEPTUAL_REFORMULATION, QUESTION_DECOMPOSITION]
- "confidence": Float between 0.0 and 1.0

Original query: {query}
"""
        try:
            raw_variants = await self.provider.generate_variants(prompt, self.max_variants)
        except Exception:
            return [original_variant]

        # Validation & deduplication
        seen = {query.lower()}
        valid_variant_strs = []
        variant_objs = []
        
        for v in raw_variants:
            v_query = str(v.get("query", "")).strip()
            if not v_query or len(v_query) > 200 or v_query.lower() in seen:
                continue
            seen.add(v_query.lower())
            valid_variant_strs.append(v_query)
            variant_objs.append(v)
            if len(valid_variant_strs) >= self.max_variants:
                break
                
        # Drift Filtering
        drift_results = self.drift_filter.filter(query, valid_variant_strs)
        
        cache_payload = {"variants": []}
        accepted_variants = [original_variant]
        
        for (v_str, sim, is_accepted), v_obj in zip(drift_results, variant_objs):
            v_data = {
                "query": v_str,
                "type": v_obj.get("type", "UNKNOWN"),
                "confidence": v_obj.get("confidence", 0.9),
                "similarity": sim,
                "is_accepted": is_accepted
            }
            cache_payload["variants"].append(v_data)
            
            if is_accepted:
                v_id = hashlib.md5((query + v_str + self.version).encode()).hexdigest()[:8]
                accepted_variants.append(QueryVariant(
                    variant_id=v_id,
                    query=v_str,
                    variant_type=v_data["type"],
                    generator=self.generator,
                    generator_version=self.version,
                    confidence=v_data["confidence"],
                    is_original=False,
                    created_at=datetime.now(timezone.utc)
                ))
                
        self.cache.set(cache_key, query, cache_payload)
        
        return accepted_variants[:self.max_variants + 1]
