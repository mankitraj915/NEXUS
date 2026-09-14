    # NEXUS Query Intelligence & Routing (Phase 6)

    ## 1. Overview
    The Adaptive Query Intelligence layer is designed to dynamically determine the optimal retrieval strategy for each search query. Instead of rigidly applying a fixed `BM25=100 / Dense=100` retrieval depth to all queries, this layer uses deterministic, low-latency feature extraction to adjust strategies, reducing unnecessary computation while maintaining or improving retrieval quality (NDCG@10).

    ## 2. Feature Extraction (Deterministic)
    The query analyzer extracts continuous features bounded in the range `[0, 1]`.

    - **`token_count`**: Total number of tokens.
    - **`unique_token_ratio`**: Diversity of vocabulary in the query.
    - **`identifier_density`**: The proportion of tokens that match explicit structural patterns (CamelCase, snake_case, version strings, file paths, acronyms).
    - **`quoted_term_density`**: The proportion of tokens explicitly wrapped in quotes.
    - **`term_rarity_score`**: A corpus-derived signal. Computed by looking up query terms in the `Term` table and calculating normalized Inverse Document Frequency (IDF). High values indicate the query is driven by highly specific corpus terms.
    - **`lexical_specificity_score`**: An aggregate score combining identifier density, quoted density, and term rarity.
    - **`natural_language_score`**: Evaluates the presence of WH-words (who, what, when, where, why, how), English stopwords, and question punctuation.
    - **`complexity_score`**: Evaluates total token count alongside conjunctions (and, or, vs, compare).

    *Note: Arbitrary token-shape heuristics like `average_token_length` were intentionally excluded to prevent false signals.*

    ## 3. Query Classification (Multi-Label)
    Continuous features are mapped into non-exclusive boolean classes.

    - **`EXACT_TERM`**: Triggered by heavy quoting or short queries with high identifier density.
    - **`IDENTIFIER_HEAVY`**: Triggered by high identifier density.
    - **`TECHNICAL`**: Triggered by high `term_rarity_score` or high `lexical_specificity_score`.
    - **`NATURAL_LANGUAGE_QUESTION`**: Triggered by a high `natural_language_score`.
    - **`CONCEPTUAL`**: Queries with moderate natural language signals but very low lexical specificity.
    - **`AMBIGUOUS`**: Very short queries lacking explicit technical or semantic structure.
    - **`MIXED`**: The default fallback class, or applied when a query is highly technical but phrased as a full natural language question.

    ## 4. Routing Policy & Thresholds
    The planner outputs a `RetrievalPlan` detailing `bm25_depth`, `dense_depth`, and `rrf_k`. 

    **Current V1 Policy:**
    1. **Lexical Dominant**: `[EXACT_TERM | IDENTIFIER_HEAVY]` -> `BM25=100, Dense=25`.
    2. **Semantic Dominant**: `[NATURAL_LANGUAGE_QUESTION | CONCEPTUAL]` (And not TECHNICAL) -> `BM25=25, Dense=100`.
    3. **Ambiguous**: `[AMBIGUOUS]` -> `BM25=100, Dense=100`.
    4. **Mixed/Fallback**: -> `BM25=100, Dense=100`.

    *Thresholds:*
    - High Lexical Specificity > 0.5
    - High Identifier Density > 0.3
    - High Natural Language Score > 0.5

    ## 5. Fallback Mechanism
    If any stage of the Adaptive routing pipeline fails (e.g., database connection drops during term rarity lookup, or planner raises an exception), the orchestrator automatically intercepts the failure, logs the error, and executes the search using the **Fixed Production Baseline (hybrid_fixed)**. The user search will never fail due to routing intelligence.

    ## 6. Benchmarks & Regret Analysis
    (To be updated with benchmark results)

    ## 7. Limitations
    - **Corpus Dependence**: Term rarity is dependent on the `ProjectStats` and `Term` tables being successfully populated by the text analyzer.
    - **Multilingual Support**: Natural language detection relies on English stopwords and WH-words. It will default to `MIXED` for other languages.
    - **No Query Rewriting**: The query intelligence layer currently only dictates retrieval candidate *depth*, but does not rewrite, expand, or multi-hop the query itself.
