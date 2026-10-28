---
name: reproduce-benchmark
description: Re-measure NEXUS retrieval quality on BEIR SciFact and compare against the recorded reports
argument-hint: "[test|train|both] [--with-rerank]"
disable-model-invocation: true
---

Reproduce the SciFact evaluation for: $ARGUMENTS (default: both splits, no rerank).
All commands run from `backend/`.

1. **Decide whether a run is needed.** Read `conditions.git_commit` in
   `docs/benchmark-evaluation-<split>.json`, then run
   `git diff --stat <commit> HEAD -- app/domains scripts/benchmark pyproject.toml uv.lock`.
   If nothing retrieval-relevant changed, cite the existing reports and stop.
2. **Make the result attributable.** Commit (or ask about) pending changes first; a report from
   a dirty tree is marked `-dirty`.
3. **Check resources.** Confirm Postgres is up on 5433 and at least ~2 GB RAM is free
   (`Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory`). Do not stop other
   projects' containers or processes; if memory is short, report and stop.
4. `uv run python -m scripts.benchmark verify` (must pass 5/5).
5. `uv run python -m scripts.benchmark prepare --dataset scifact`. It re-ingests only if the
   manifest's project is missing or incomplete (about 15 minutes on CPU).
6. Run each split in the background with a log file and watch for `NDCG@10:` lines:
   `uv run python -m scripts.benchmark evaluate --dataset scifact --split <split> [--with-rerank]`.
   Running it overwrites that split's report, so keep the previous JSON via `git show HEAD:<path>`.
7. **Compare** new vs previous JSON per configuration. For unchanged code and model revisions,
   NDCG@10/Recall/MRR should match to 4 decimals; explain any difference before using it.
   Treat latency differences as noise unless they cross retriever families.
8. Update README / `docs/benchmarking.md` only with numbers from the new reports, labelling the
   300-query split as tuned and the 809-query split as untuned.
