# What

<!-- One or two sentences: what changes for a user of the library? -->

# Why

<!-- The problem this solves. Link the issue if there is one. -->

# How it was verified

<!-- Paste the actual output, not "tests pass". -->

```
ruff check src/ tests/ examples/ benchmarks/ scripts/
mypy src/agentflow/
pytest                      # includes the 90% coverage gate
```

# Checklist

- [ ] Lint, typecheck, and the full suite are green locally (output pasted above)
- [ ] New behaviour has tests — happy path **and** at least one failure path
- [ ] `CHANGELOG.md` `[Unreleased]` updated
- [ ] Public API change? `PUBLIC_API.md` updated, and a deprecation shim if it breaks
- [ ] Hard-to-reverse decision? An ADR added under `docs/adr/`
- [ ] Docs updated (`README.md`, `docs/`) if commands, config, or behaviour changed
- [ ] No secrets, debug leftovers, or unrelated drive-by changes in the diff
