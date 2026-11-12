# 0004 - Side-effecting tools run at most once per run, through an effect ledger

Date: 2026-11-11 · Status: accepted with amendments (2026-11-12)

## Context

`Pipeline(retry_failed_agents=N)` retries a failed agent by calling
`agent.execute()` again from the top. Every tool call the first attempt made is
made again. For a read tool that costs tokens. For a write tool it repeats the
effect.

Reproduced on 0.7.1: an agent calls `charge_card(customer="ali", amount=50)`,
the next LLM call fails twice, the pipeline retries the agent, and the card is
charged twice. `output: done | charges executed: 2`.

The in-loop duplicate guard in `AgentSpec._run_tool_loop` does not help: it only
sees calls inside one attempt.

The research says this class of bug is general. "When Tool Calls Succeed but
Workflows Fail" (arXiv 2609.15397) catalogs eight external-effect anomalies
(missing, duplicated, surviving-aborted effects, ...) and finds that none of the
capabilities needed to exclude them is fully expressible in MCP tool
annotations across 98,291 registered tools. RetryBench (arXiv 2610.15884) shows
models misuse idempotency keys depending on how a key is described. Leaving the
decision to the model is not safe.

## Decision

1. A tool opts in with `@tool(side_effect=True)`. Default `False`: read tools
   behave exactly as today, with no ledger traffic.
2. `Pipeline(effect_ledger=...)` turns protection on. No ledger, no change.
3. Each side-effecting call gets a key:
   `sha256(json([run_id, agent_name, tool_name, validated_args], sort_keys))`.
   Validated args, so `{"amount": "50"}` and `{"amount": 50}` collide when the
   schema coerces them.
4. Before running, the call **claims** the key. The ledger answers atomically:
   - no record: caller owns it, record is `PENDING`, the tool runs;
   - `DONE`: the stored result is returned, the tool does not run;
   - `FAILED`: reclaimed, the tool runs again (the tool itself reported failure);
   - `PENDING`: someone started this effect and never finished. The library
     cannot know whether it happened, so it raises `EffectUncertainError`,
     which the pipeline does not retry. A human decides.
5. After running: success settles `DONE` with the stringified result; a raised
   exception settles `FAILED`; cancellation (timeout, crash) leaves `PENDING`.
6. Ledger errors fail closed: if `claim` raises, the tool does not run.
7. `run_id` widens from 8 to 32 hex characters. With a durable ledger, an 8-char
   id collides at roughly 65k runs (birthday bound) and a collision would replay
   another run's result.

Backends: `InMemoryLedger` (one process, covers in-run retries) and
`SqliteLedger` (stdlib `sqlite3`, survives restarts and covers `resume()`).
No new runtime dependency.

## Consequences

- Retries and `resume()` stop repeating writes when a ledger is configured.
- A crash between "effect happened" and "settled" surfaces as
  `EffectUncertainError` instead of a silent duplicate. That is the honest
  answer: only the tool's owner can check the outside world.
- Two different agents making the same call in one run are two effects (the
  agent name is in the key). Intended.
- The same call in two runs is two effects (the run id is in the key). Intended:
  `serve()` triggers are separate runs.
- Compensation (undoing an effect after a later failure) is out of scope. It
  needs per-tool undo semantics and is a separate decision.
- `Swarm` and `swarm_routing` call `Tool.acall` directly and are not covered.

## Amendments (2026-11-12)

Adopted for 0.8.0 with three changes, from the experiments that ran alongside it
(`agentflow-lab`, EXP-B B1 on real ROS 2 with a real model):

1. **The key includes the occurrence number.** With `(run_id, agent, tool, args)` alone, the
   second identical command in a run replays the first one's result and does not act: a
   real model sent the same `move` twice in 3 of 10 runs, and both were meant. The key is
   `sha256([run_id, agent, canonical(tool, validated_args), n])`, where `n` counts earlier
   calls with the same tool and validated arguments in the agent's current conversation.
   A retry starts a new conversation, so its first call is `n=0` again and replays; a
   resume continues the saved conversation, so the count continues (it is rebuilt from the
   trace). Write calls run one at a time, so the count follows the model's order.
2. **The flag is `effect="write"`** from ADR 0006, not `side_effect=True`.
3. **The ledger is always on for write tools.** Without `effect_ledger=`, each run gets its
   own `InMemoryLedger`, which covers retries in one process and is freed with the run. A
   durable ledger is needed for `resume()` in a new process; a pipeline with a memory
   backend and write tools but no durable ledger warns once.

`EffectUncertainError` is exported from `agentflow` (it is raised out of `Pipeline.run`).
The ledger classes stay in the opt-in module `agentflow.effects`. Using the same claim for
HITL `resume()` (HITL-2, HITL-3) is the next step of 0.8.0.
