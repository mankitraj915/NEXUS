---
name: verify-release
description: Run the full NEXUS pre-push verification and publication audit, then push safely
disable-model-invocation: true
---

Verify the release on the current branch. Report each step's real result; stop at the first
failure and fix the cause rather than weakening a check. Commands run from `backend/`.

1. **Scope.** `git status --short` and `git log --oneline @{u}..HEAD` (or the full log if there is
   no upstream). Stage only intended paths; never `git add -A`.
2. **Fast checks:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy .`,
   `uv run python -m scripts.benchmark verify`.
3. **Integration** (Postgres up): `uv run alembic upgrade head`, `uv run pytest tests/ -rs` with
   zero skips, then `uv run python -m scripts.dev_smoke --seed`.
4. **Claims.** Every number in `README.md` and `docs/` must match
   `docs/benchmark-evaluation-*.json`; flag anything unsourced.
5. **Publication audit.** Over every commit that will be pushed (not just the tree):
   - secrets: `git log -p <range> | grep -nE "sk-|ghp_|github_pat_|hf_|AKIA|PRIVATE KEY|password\s*="`
   - personal data and local paths (phone numbers, home directories)
   - files no longer tracked but still in history: compare `git log --name-only` against `git ls-files`
   Anything found blocks the push until the user decides.
6. **Review (non-trivial diffs).** Run `/code-review` on the range and address real defects.
7. **Push** only with the user's go-ahead for this destination. Never force-push. Confirm with
   `git ls-remote origin <branch>` matching `git rev-parse HEAD`, then report the CI result from
   `gh run list --limit 1` / `gh run watch`.
