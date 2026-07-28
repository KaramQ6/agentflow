# 0003 — An unpriced model reports zero cost, loudly

Date: 2026-07-29 · Status: accepted

## Context

`estimate_cost()` returns `0.0` for any model without an entry in the pricing
table. The table shipped with OpenAI and Anthropic models only, while the
README quickstart points users at Groq. Every run on the provider the docs
recommend therefore reported `total_cost: $0.000000` — indistinguishable from a
genuinely free local model, and wrong in the one number the library claims as
its differentiator.

Options: raise on an unknown model; guess a default price; or keep returning
zero but stop being silent about it.

Raising is wrong — self-hosted models really are free, and a cost table has no
business breaking a working pipeline. Guessing is worse than zero: a fabricated
number is harder to notice than an obviously absent one.

The deeper problem is that any table baked into a release is stale on contact.
Providers change prices; a bigger table postpones the failure without fixing
its shape.

## Decision

Unknown models keep costing `0.0`, and the placeholder is made visible in two
places:

- `warn_if_unpriced()` logs once per process, from the LLM client where a model
  id first arrives, naming the model and the `register_price()` call that fixes
  it.
- `PipelineResult.unpriced_models` lists every model in the run whose cost is a
  placeholder. A non-empty list means `total_cost` is an undercount.

`estimate_cost()` itself stays a pure function; the logging lives in a separate
function called from the effectful edge.

The table is expanded to cover the providers the docs recommend, carries a
review date, and documents `register_price()` as the authoritative override
rather than the fallback.

## Consequences

Easier: a user on an unpriced model finds out from a log line and a result
field instead of believing a wrong number. Pipelines never break because of a
pricing gap. A test asserts the models named in the docs are priced, so the
quickstart cannot silently regress to $0.

Harder: `total_cost` is still an undercount when a model is unpriced —
`unpriced_models` tells you, but you have to look. The bundled prices remain
indicative and will drift; they are documented as such rather than guaranteed.

Exit path: if the drift becomes a real support burden, the table moves to a
data file refreshed independently of releases. The `is_priced` /
`warn_if_unpriced` / `unpriced_models` surface is unaffected by where the
numbers come from.
