# 0002 — Structured output by prompt injection and repair, not provider JSON modes

Date: 2026-07-29 · Status: accepted

## Context

`output_schema` validated an agent's reply against a Pydantic model and did
nothing else. The model was never told what shape to produce — the README's own
example worked around this by embedding `Model.model_json_schema()` into the
user prompt by hand — and a single malformed reply raised
`AgentOutputValidationError`, killing the run.

Three ways to close that gap:

1. Send the schema to the provider as a native `response_format` /
   `json_schema` parameter, which constrains decoding server-side.
2. Put the schema in the prompt and validate what comes back.
3. Both, with a runtime capability check per provider.

agentflow's compatibility claim is "any OpenAI-compatible endpoint" — OpenAI,
Groq, Together, OpenRouter, Ollama, vLLM. Native structured-output support
across that set is uneven: differently named, differently shaped, differently
strict, and absent entirely on older self-hosted stacks. Option 1 would make
the headline claim conditional and fail at runtime on endpoints the README
tells people to use. Option 3 means maintaining a capability matrix of other
people's servers, which is exactly the ecosystem-sized commitment this library
exists to avoid.

## Decision

Option 2. The JSON Schema is appended to the system prompt — last, so the
format contract is the final thing the model reads — and the reply is validated
locally. On a validation failure, the model is shown its own output and the
validation errors and asked to correct itself, bounded by `output_retries`
(default 1).

A reply wrapped in a markdown code fence is unwrapped locally instead of
spending a repair round-trip on a formatting artefact. Nothing further is
guessed at: digging a JSON object out of surrounding prose is how you silently
validate the wrong thing.

Callers who *do* want a provider's native JSON mode pass it themselves through
`LLM.generate(**extra)`, which forwards any keyword straight to the provider.
agentflow never sets it.

## Consequences

Easier: `output_schema` works the same way on every OpenAI-compatible endpoint,
including local ones. Transient formatting failures stop killing runs. Users no
longer hand-write schemas into prompts.

Harder: schema instructions consume prompt tokens on every call, and a repair
is a second billed call. Both are counted into the agent's usage and the
pipeline budget rather than hidden, but they are real cost. Prompt-level
constraint is also weaker than constrained decoding — a determined model can
still return prose, which is what the repair round is for.

Exit path: adding native `response_format` later is additive — an opt-in
argument on `@Agent` that sets what `**extra` already forwards. The prompt path
stays as the portable default, so nothing has to be removed.
