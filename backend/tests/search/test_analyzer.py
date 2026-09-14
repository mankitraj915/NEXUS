import pytest
from app.domains.search.analyzer import analyze_text

def test_analyze_basic():
    tokens = analyze_text("PostgreSQL 17 supports AI/ML workloads.")
    assert tokens == ["postgresql", "17", "support", "ai", "ml", "workload"]
    
def test_analyze_stopwords():
    tokens = analyze_text("The quick brown fox jumps over the lazy dog.")
    assert tokens == ["quick", "brown", "fox", "jump", "lazi", "dog"]
    
def test_analyze_no_stopwords_removal():
    tokens = analyze_text("to be or not to be", remove_stopwords=False)
    assert tokens == ["to", "be", "or", "not", "to", "be"]
    
def test_analyze_unicode_and_hyphens():
    tokens = analyze_text("AI-native state-of-the-art naïve")
    assert tokens == ["ai-nat", "state-of-the-art", "naïv"]
