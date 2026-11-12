# 0006 - Tools declare their effect: read, write, or unset

Date: 2026-11-12 · Status: accepted · Related: [0004](0004-effect-ledger-for-side-effecting-tools.md)

## Context

The tool loop treated every tool the same way, and each choice was wrong for some tool:

- A duplicate-call guard rejected a repeated call with the same arguments. For a sensor
  read that is a normal request (LOOP-1: 33 rejections in 5 real robot runs). For a motion
  command it is a refused order (NEW-14: "forward 1 m, again" rejected 5 times in 10 runs).
- Calls in one model reply ran concurrently. For two motion commands on one topic that
  interleaved contradictory commands (NEW-12, 5/5 runs on real ROS 2).
- `retry_failed_agents` re-ran the whole agent, so a write repeated (ADR 0004).

The library cannot infer what a tool does. Two booleans (`read_only`, `side_effect`) allow
a contradictory pair. One field with three states says it once.

## Decision

`effect: Literal["read", "write"] | None = None` on `tool()` and `Tool`.

| Behaviour | `"read"` | `"write"` | unset (`None`) |
|---|---|---|---|
| Runs concurrently with other calls in a reply | yes | no: one at a time, in the model's order | yes (unchanged) |
| Duplicate-call guard | off | off | on (unchanged) |
| Effect ledger (ADR 0004) | no | yes, keyed by occurrence | no |
| Agent may be retried after running it | yes | yes, the ledger replays it | **no** |
| Sync function allowed | yes | **no**, `TypeError` at decoration | yes |

Why unset stays concurrent and guarded: existing pipelines keep their speed and behaviour.

Why unset blocks retries: a silent duplicate (an email sent twice, a card charged twice, a
robot moved twice) is the worst outcome the library can produce, and it cannot tell whether
an unset tool is that kind. The error names the tools and says to set `effect`. This is a
behaviour change for `retry_failed_agents` users whose agents use unset tools.

Why write tools must be async: a sync function runs in a thread that cannot be cancelled,
so it keeps acting after a timeout (NEW-07: 2.5 s of motion after the timeout on real ROS 2).

## Consequences

- Robot and side-effect tools get the right semantics by declaring one word.
- Tools without a declaration behave as in 0.7, except that their agents are not retried.
- Write calls in one reply are slower than before, because they no longer overlap. Reads are
  unaffected.
- `Swarm` and `SupervisorAgent` workers call `Tool.acall` directly, outside the tool loop, so
  serialization and the ledger do not cover them yet.
