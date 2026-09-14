import numpy as np
from typing import List, Tuple
from app.domains.embeddings.pipeline import provider

class SemanticDriftFilter:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        a = np.array(vec1)
        b = np.array(vec2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def filter(self, original_query: str, variants: List[str]) -> List[Tuple[str, float, bool]]:
        """
        Takes the original query and generated variant strings.
        Returns a list of (variant, similarity_score, is_accepted)
        """
        if not variants:
            return []
            
        # We can embed the original and all variants in one batch for efficiency
        texts_to_embed = [original_query] + variants
        embeddings = provider.embed_documents(texts_to_embed)
        
        orig_emb = embeddings[0]
        variant_embs = embeddings[1:]
        
        results = []
        for variant, v_emb in zip(variants, variant_embs):
            sim = self._cosine_similarity(orig_emb, v_emb)
            is_accepted = sim >= self.threshold
            results.append((variant, sim, is_accepted))
            
        return results
