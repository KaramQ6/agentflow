# agentflow — Level-Up Analysis

Reviewer stance: skeptical maintainer deciding whether a real team should depend on this.
Ordered by impact-per-effort. Verified against the code at v0.5.0 (working tree), not the README's claims.

---

## The 3 things that matter most

**1. The library has an identity crisis, and it's visible from the outside.**
The README says *"agentflow is a deliberately narrow library, not a framework"* — while
`__init__.py` exports `DockerSandbox`, `SubprocessSandbox`, `MQTTTrigger`, `SupervisorAgent`,
`VectorContext`, and an HITL pause/resume state machine. Git history shows a C++ DAG engine
added in one release ("pillar1") and deleted in the next, plus `swarm_routing.py` and
`distillation.py` that aren't even exported. A reviewer evaluating this for production reads
that history and concludes: *the maintainer doesn't know what this is yet, so I can't depend
on it.* Nothing else in this document matters until the scope is actually narrow, not
narrated as narrow.

**2. The string-only data plane is the one hard-to-reverse API mistake, and it's still reversible today.**
Agents communicate through `context: dict[str, str]`. `Pipeline.run()` takes a bare `str`.
`LLM.generate()` returns an untyped `dict[str, Any]`. Structured output (`output_schema`)
is validated and then **buried in `metadata["validated_output"]`** while downstream agents
receive the raw JSON string. For a library whose headline trust signal is "fully typed,
`mypy --strict`", the actual data flowing between agents is untyped strings. PydanticAI's
entire pitch is typed agent I/O — this is the exact axis you'll be compared on. Fix it
before 1.0 or never.

**3. Trust hygiene is broken in ways that take an hour to fix.**
`src/agentflow/__init__.py` says `__version__ = "0.3.0"`; `pyproject.toml` says `0.5.0`.
The import name is `agentflow` but the pip name is `agentflowkit` — every new user's first
five minutes includes "why doesn't `pip install agentflow` work". Commit messages like
"massive architecture upgrade" between minor versions signal churn. These are trivial fixes
that currently scream *hobby project* to anyone doing due diligence.

---

## Analysis 1 — Positioning & the "why not X" test

**The honest niche:** agentflow is a *readable, two-dependency, cost-aware parallel DAG
runner for OpenAI-compatible endpoints*. Its defensible claim is auditability: the core
(pipeline + agent + llm + tools + types, ~1,700 lines) can be read in an afternoon,
type-checks under `mypy --strict`, and has retries, timeouts, caching, cost, and events
built in rather than as plugins. That is a real niche — teams burned by LangChain's
dependency graph, running on Groq/Ollama/vLLM, who would otherwise hand-roll
`asyncio.gather` plus retry/cost boilerplate.

**The single sharpest wedge:** **cost-aware parallel orchestration on a core you can audit.**
Nobody else owns "every run tells you what it cost, per agent, with cache hits billing $0,
in a library small enough to read before deploying." LangGraph won't be small. CrewAI won't
be rigorous. PydanticAI won't center DAG parallelism. Own this one thing; everything else
(swarm, HITL, sandbox, MQTT) is secondary or actively dilutive.

**Who should NOT use agentflow (say this in the README):**
- Anyone already inside LangChain/LlamaIndex — the ecosystem gravity isn't worth fighting.
- Anyone needing durable workflows that survive process death (the HITL resume is
  memory-backed state, not a workflow engine — don't pretend otherwise).
- Anyone needing native (non-OpenAI-compatible) SDKs: Bedrock, Vertex native.
- Anyone wanting RAG batteries: loaders, splitters, vector-store integrations.
- Anyone whose graph is dynamic per-run (agentflow's DAG is static once built).

---

## Analysis 2 — API design critique

### 2a. Hard to reverse once adopted — fix NOW

**The untyped data plane** (the #2 issue above). Concretely:

```python
# Before — llm.generate returns a dict users index blindly:
response = await llm.generate(messages)
content = response["content"]          # typo = runtime KeyError
cost = response.get("cost", 0.0)       # every caller re-invents the default

# After — a frozen model; attribute access, mypy-checked:
class LLMResponse(BaseModel):
    content: str
    tokens: TokenUsage
    cost: float
    model: str
    cached: bool = False
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None

response = await llm.generate(messages)
response.content
```

```python
# Before — structured output validated, then thrown away for downstream agents:
context["analyst"]                                    # raw JSON string
result.get("analyst").metadata["validated_output"]    # the actual data, buried

# After — validated output IS the agent's output downstream:
@Agent(name="analyst", role="...", output_schema=Report)
...
report: Report = context["analyst"]    # context: dict[str, Any]
```

`context: dict[str, str]` → `dict[str, Any]` is the enabling change. Do it while the user
count is small; after adoption it's a breaking change to every agent body ever written.

**Mutable shared agent state — this is a live bug, not just a smell.**
`@Agent` produces a module-level singleton. `Pipeline._execute_node()` calls
`agent.set_session(session_id)` and `agent.set_approval_policy(...)` — mutating that
singleton. Two pipelines (or two `serve()` requests — `serve` explicitly runs concurrent
`run()`s) sharing one agent will cross-contaminate sessions: pipeline A's agent reads and
writes pipeline B's memory. The fix is to stop mutating and pass run-scoped state through
the call:

```python
# Before (pipeline.py:172-177) — mutate the shared instance:
agent.set_session(session_id)
agent.set_approval_policy(policy)
result = await agent.execute(task, context, self._llm)

# After — run-scoped context travels with the call:
result = await agent.execute(task, context, RunContext(llm=self._llm,
                                                       session_id=session_id,
                                                       approval_policy=policy))
```

This also fixes `BaseAgent.execute(task, context, llm)`'s signature being frozen at three
positional params — `RunContext` is the extension point so the signature never breaks again.

**`Agent(...)` returning the private `_DecoratorAgent`.** The decorator's return type is a
private class users hold references to, pass to `Pipeline.add`, and will inevitably
introspect. Rename it public (`AgentSpec` or fold into `BaseAgent`) before someone
depends on the underscore name.

**`hasattr`-based duck typing** (`hasattr(agent, 'set_session')`,
`hasattr(node.agent, 'resume_execution')` in pipeline.py) — invisible contracts. A user
subclassing `BaseAgent` gets silently different behavior depending on which optional
methods they happened to define. Replace with an explicit `Protocol` or move the
capability into `BaseAgent` with default implementations.

### 2b. Semantics that will surprise users

- **`PipelineResult.output` = "last agent's output"** — undefined when the final level has
  multiple parallel agents; it's whichever ran last in list order. Either make it the
  outputs of all sink nodes, or document that it's only meaningful for single-sink DAGs
  and raise/warn otherwise.
- **`total_duration` sums per-agent durations** — in a *parallelism* library, the headline
  duration metric double-counts parallel time. Rename to `agent_seconds` and add
  `wall_time` measured around the run. This one is embarrassing given the pitch.
- **`condition` receives the full accumulated context; `execute` receives dep-scoped
  context.** Two different context shapes for the same node. Pick one (dep-scoped is the
  right one) — it changes what user lambdas can see, so it's a breaking change later.
- **The system prompt is hardcoded**: `f"You are a {role}. Provide clear, thorough,
  well-structured responses."` — unoverridable trailing instruction injected into every
  agent. Add `system_prompt=` to `@Agent`.
- **Memory injection silently truncates to 300 chars** (agent.py:110) — silent data loss
  with no knob and no log line.
- **`status: str`** on `PipelineResult` → `Literal["completed", "paused"]`.
- **`depends_on` forbids forward references** (dep must be added first). Fine as a choice,
  but it forces users to topologically sort their own code; validating at `run()` instead
  would cost nothing.

### 2c. What's genuinely good — keep and defend
Dep-scoped context (agents only see declared dependencies) is a *great* decision most
competitors don't make. Kahn's-levels parallelism is simple and explainable. The tool loop
(dedup, truncation, sliding window, concurrent tool calls) is careful. Exceptions carry
agent names. `stream()` cancels the background task when the consumer abandons the
generator. This core deserves the audit-friendly pitch — the periphery undermines it.

---

## Analysis 3 — The adoption gap (scored 1–5)

| Dimension | Score | Justification | Action to raise it |
|---|---|---|---|
| Documentation | 3 | mkdocs + guides exist; but exports like `SupervisorAgent`, `sandboxed_tool`, `MQTTTrigger` are thinly documented, no migration notes between 0.x releases | Rule: every name in `__all__` has a reference page and a runnable snippet, or it gets un-exported. Add UPGRADING.md per minor. |
| Reliability signals | 3 | CI matrix 3.10–3.12, 90% coverage gate, mypy strict — good. But `__version__` mismatch, deleted-feature churn, no 3.13 | Fix version (single-source from `importlib.metadata`), add 3.13, tag every release, keep CHANGELOG honest about removals. |
| Provider coverage | 2.5 | "OpenAI-compatible" is a fine stance but untested as a claim — nothing in CI exercises Groq/Ollama/OpenRouter quirks (e.g. tool-call format drift) | A recorded-response (VCR-style) provider matrix in CI + a documented compat table: provider / tools / streaming / tested version. |
| Observability & debuggability | 3 | Hooks + event stream are real; but no way to see *why* the DAG resolved the way it did, or why a condition skipped an agent | `pipe.explain()` — dry-run printing resolved levels, dependencies, and condition outcomes; include skip reasons in `agent_skipped` events. |
| Error messages | 4 | Exceptions are typed, carry agent/tool names, cycle detection exists | Cycle error should name the cycle members, not just "Cycle detected". |
| Ecosystem fit | 2 | README name-drops OpenTelemetry/Langfuse but ships no adapter; hooks are the right seam, unused | Ship `agentflow.contrib.otel.OTelHooks` (~50 lines) — one import to spans. Highest fit-per-line item on this list. |
| Trust signals | 2 | MIT clear, changelog exists — but version mismatch, pip-name/import-name split, sandbox-in-core security surface, feature churn | Fix version; add SECURITY.md; publish a stability policy (below); stop shipping code-execution sandboxes in the "minimal" core package. |

The sandbox deserves its own sentence: **a library advertising "minimal, auditable, two
dependencies" ships 461 lines of Docker/subprocess arbitrary-code execution in core.**
That's the single largest security surface in the package, in the component least related
to the wedge. Move it to a separate package or clearly-marked extra.

---

## Analysis 4 — The killer example

The wedge is *visible cost + visible parallelism*. Design the demo so both are undeniable
in one run, with zero API keys required to try it (Groq free tier or Ollama).

**"Earnings-call triage"** — realistic diamond DAG with genuine dependencies:

```
                    ┌─ financials_analyst ──┐
transcript_fetcher ─┼─ sentiment_analyst  ──┼─ risk_synthesizer ─ brief_writer
                    └─ competitor_scanner ──┘
```

- `transcript_fetcher` uses **tools** (fetch + chunk) — shows the ReAct loop.
- The three middle analysts run **in parallel** — the event stream prints them starting
  simultaneously with live `agent_complete` lines.
- `risk_synthesizer` has an `output_schema` — shows typed output.
- The script ends by printing: **wall time vs. sequential estimate** (e.g. "11.2s
  parallel, ~29s sequential") and **per-agent + total cost**.
- Then it runs a second time with `InMemoryCache` and prints the delta:
  "$0.0041 → $0.0007, 3 cache hits." That line is the whole pitch in one run.

Build it as `examples/earnings_triage.py`, make its *actual captured output* the first
code block in the README after install, and record a 20-second asciinema of it. Every
current README example is either a toy ("multiply two numbers") or unverifiable prose;
this replaces all of them as the centerpiece.

---

## Analysis 5 — Path to real dependents (6 months)

**The 3–5 guarantees worth depending on:**
1. **Typed data plane end-to-end** (LLMResponse model, `Any` context, schema output flows
   downstream) — the PydanticAI-parity feature.
2. **A written stability contract**: `PUBLIC_API.md` listing exactly what's covered by
   semver, a deprecation policy (warn one minor, remove next major), and explicit 1.0
   criteria. Teams adopt contracts, not features.
3. **Provider compat matrix tested in CI** — turns "OpenAI-compatible" from a hope into
   a guarantee.
4. **OTel hooks adapter** — makes it composable with whatever observability stack the
   team already runs.
5. **Cost budgets**: `Pipeline(budget_usd=0.50)` that aborts the run when exceeded.
   Nobody else has this as a first-class primitive and it's ~40 lines on top of the
   existing cost plumbing. This is the wedge feature.

**What hitting 1.0 requires:** the scope cut executed (below), the typed data plane done,
zero known breaking changes queued, three consecutive minors with no public-API breaks,
and the version/naming hygiene fixed. Realistically: 1.0 in month 5–6, not sooner.

**Where first users come from:** the cost-conscious OpenAI-compatible crowd — r/LocalLLaMA,
Groq and Ollama Discords, a Show HN. These communities *cannot* use most heavyweight
frameworks comfortably (local endpoints, tight budgets) and are exactly who the wedge
serves. What gets them to try it: the killer example running against Ollama with zero
API keys, and the cost-budget primitive.

**What to explicitly NOT build (scope creep already killing this):**
- No RAG: loaders, splitters, embeddings, vector stores. `VectorContext` is already over
  the line — freeze it or extract it.
- No C++/Rust engine. The DAG resolution is microseconds against multi-second LLM calls;
  it was rightly deleted, don't bring it back.
- No agent marketplace, personas, or prompt-template library.
- No general workflow engine (durable execution, cron, queues). `serve()` + MQTT is
  already drifting there — extract triggers to `agentflowkit-triggers`.
- Sandbox → separate package (`agentflowkit-sandbox`) with its own security policy.
- Swarm/`SupervisorAgent`: keep only if it stays ≤ its current size; `swarm_routing.py`
  and `distillation.py` are unexported speculative code — delete or ship, don't carry.

---

## Analysis 6 — Competitive teardown

**LangGraph** — Better: durable checkpointing/persistence, human-in-the-loop that survives
process death, LangSmith tracing, Studio, massive ecosystem and mindshare. Worse: dependency
weight, conceptual overhead (channels, reducers, compiled graphs) for what is often a
5-node DAG, hard to audit. **Structural gap agentflow owns:** LangGraph can never be
small — its value *is* the ecosystem. "Read the whole orchestrator before prod" is
permanently unavailable to them.

**CrewAI** — Better: marketing, onboarding, templates/personas, huge top-of-funnel. Worse:
magic-heavy abstractions, weak typing, unpredictable token burn, hard to reason about
execution order. **Gap:** engineering rigor + cost transparency. CrewAI's audience is
prototypers; it structurally won't chase `mypy --strict` teams, and per-agent USD
accounting undercuts them where they're weakest.

**PydanticAI** — the dangerous one. Better: typed agent I/O done right, real multi-provider
abstraction (native Anthropic/Gemini/Bedrock), the Pydantic team's trust halo, growing
fast. Worse: orchestration is not its center of gravity — multi-agent parallel DAG
composition means bolting on pydantic-graph, which is heavier and less legible than
`pipe.add(x, depends_on=[...])`. **Gap:** declarative parallel DAG + cost/budget
primitives. But note: if agentflow doesn't fix its typed data plane, PydanticAI wins the
comparison on agentflow's own claimed strength. This is why Analysis 2a is urgent.

**Raw asyncio** — Better: zero dependencies, zero abstraction tax, infinitely flexible.
Worse: every team re-writes retries, backoff-with-Retry-After, cost tables, cache keys,
event streams, timeout handling — badly, under deadline. **Gap:** agentflow's honest
one-line pitch against DIY: *"the 1,700 lines you were going to write around
`asyncio.gather` anyway, already typed and tested."* This baseline, not LangGraph, is the
real competitor for the target user — position against it explicitly.

---

## Next 2 weeks — concrete actions

1. **Fix version hygiene** (1 hr): single-source `__version__` from package metadata;
   verify pyproject/`__init__`/git tag agree; add a CI check that fails on mismatch.
2. **Execute the scope cut** (1–2 days): remove `sandbox`, `triggers`/`serve`,
   `swarm_routing`, `distillation` from the core package (extras or delete); shrink
   `__all__` to the narrow story; update README to match reality.
3. **Fix the shared-agent mutation bug** (½ day): `RunContext` passed through `execute()`
   replacing `set_session`/`set_approval_policy` mutation; regression test with two
   concurrent `run()`s sharing an agent.
4. **Type the data plane** (2–3 days): `LLMResponse` model, `context: dict[str, Any]`,
   `output_schema` results flow to downstream agents as model instances; deprecation
   shims for the dict access.
5. **Rename `total_duration` → `agent_seconds`, add `wall_time`** (1 hr, with shim).
6. **Build `examples/earnings_triage.py`** (1–2 days) and make its real output the README
   centerpiece; record the cache-hit second run.
7. **Write `PUBLIC_API.md`** (½ day): covered surface, deprecation policy, 1.0 criteria.
8. **Ship `OTelHooks`** (½ day): one adapter class in `contrib/`, documented with a
   Jaeger screenshot.

**The blunt bottom line:** the core four files are genuinely good — better engineered than
most 0.x agent libraries. The project's risk is not code quality; it's that it behaves
like four different projects sharing a repo, with trust-hygiene cracks a reviewer spots in
minutes. Cut to the wedge, type the data plane, fix the hour-long embarrassments, and
there is a real (if narrow) adoption path. Keep accreting pillars, and PydanticAI ends the
story by default.
