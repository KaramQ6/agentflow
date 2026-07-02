"""Multi-agent memory showcase — two agents share context via InMemoryContext.

Demonstrates cross-agent memory in agentflow: a Researcher agent gathers
information and persists it to shared memory, then a Writer agent (in a
completely separate pipeline run) recalls that data purely through the
memory module — no explicit ``depends_on`` or context-passing required.

The shared ``session_id`` is the bridge: both pipeline runs reference the
same session, so the Writer's system prompt is automatically enriched with
the Researcher's previously stored output.

Run: python examples/memory_chat_agents.py
Requires GROQ_API_KEY (free at console.groq.com) or edit the api_key below.
"""

from __future__ import annotations

import asyncio
import os

from agentflow import LLM, Agent, InMemoryContext, Pipeline

# ─── LLM setup ─────────────────────────────────────────────────────────────────
# Swap these for your provider. Groq is free and fast for demos.
api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
    raise SystemExit(1)

use_groq = bool(os.environ.get("GROQ_API_KEY"))
llm = LLM(
    model="llama-3.3-70b-versatile" if use_groq else "gpt-4o-mini",
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1" if use_groq else None,
    max_tokens=768,
)


# ─── Shared memory store ───────────────────────────────────────────────────────
# InMemoryContext is an in-process dict with per-entry TTL and LRU eviction.
# It is thread-safe (asyncio.Lock) and requires no external infrastructure.
# Both pipelines below will share this exact instance.
shared_memory = InMemoryContext(default_ttl=3600, max_entries=1000)


# ─── Agents ────────────────────────────────────────────────────────────────────


@Agent(name="researcher", role="Research Analyst")
async def researcher(task: str, context: dict) -> str:
    """Research a topic and return key findings with sources.

    This agent runs first and its output is automatically persisted to the
    shared memory under the session's namespace (keyed by agent name).
    """
    return (
        f"Research the following topic thoroughly. Provide 3-5 key findings "
        f"with brief explanations and mention credible sources where possible:\n\n"
        f"Topic: {task}"
    )


@Agent(name="writer", role="Content Writer")
async def writer(task: str, context: dict) -> str:
    """Write a blog post based on the task description.

    The system prompt will automatically include any prior agent outputs
    that were saved to shared memory under the same session_id. The Writer
    therefore sees the Researcher's findings even though this agent was
    added to a separate pipeline with no dependency declaration.
    """
    return (
        f"Write a concise, engaging blog post about the following topic. "
        f"Use the research findings that were provided in the system prompt "
        f"as your primary source material. Include an introduction, 2-3 body "
        f"sections, and a conclusion.\n\n"
        f"Topic: {task}"
    )


# ─── Pipeline 1: Research phase ────────────────────────────────────────────────
# The Researcher runs alone. Its output is stored in memory under
# key "researcher" → session "demo-session" automatically.
async def phase_one(topic: str) -> dict:
    """Run the Researcher and persist findings to shared memory."""
    print(f"\n{'─' * 60}")
    print("PHASE 1 — Researcher gathers information")
    print(f"{'─' * 60}")

    # The memory and session_id are attached to the pipeline. Every agent
    # in the pipeline gets its session set before execution, so the
    # Researcher will save its output under "demo-session".
    pipe = Pipeline(
        llm=llm,
        memory=shared_memory,
        session_id="demo-session",
    )
    pipe.add(researcher)

    result = await pipe.run(topic)
    agent_result = result.get("researcher")

    print("\nResearcher output (first 500 chars):")
    print(agent_result.output[:500] + ("..." if len(agent_result.output) > 500 else ""))
    print(f"\nTokens: {result.total_tokens} | Cost: ${result.total_cost:.6f} | "
          f"Duration: {result.total_duration:.1f}s")

    return await shared_memory.load_context("demo-session")


# ─── Pipeline 2: Writing phase ─────────────────────────────────────────────────
# The Writer runs in a new pipeline but shares the same memory instance AND
# the same session_id. The Writer's system prompt will be enriched with the
# Researcher's saved output from Phase 1.
async def phase_two(topic: str) -> None:
    """Run the Writer, which recalls Researcher data from shared memory."""
    print(f"\n{'─' * 60}")
    print("PHASE 2 — Writer recalls research from memory")
    print(f"{'─' * 60}")

    # Same memory instance and session_id — the bridge between phases.
    pipe = Pipeline(
        llm=llm,
        memory=shared_memory,
        session_id="demo-session",
    )
    pipe.add(writer)

    result = await pipe.run(topic)
    agent_result = result.get("writer")

    print("\nFinal blog post:")
    print("=" * 60)
    print(agent_result.output)
    print("=" * 60)
    print(f"\nTokens: {result.total_tokens} | Cost: ${result.total_cost:.6f} | "
          f"Duration: {result.total_duration:.1f}s")


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    topic = "The impact of small language models running on edge devices in 2025"

    print("\nMulti-Agent Memory Showcase")
    print(f"Topic: {topic}")
    print(f"Provider: {'Groq' if use_groq else 'OpenAI'} · "
          f"Model: {llm.model}")
    print("Session: demo-session | Store: InMemoryContext (TTL=3600s, LRU=1000)")

    # Phase 1: Researcher runs and saves to memory.
    memory_state = await phase_one(topic)

    # Inspect the memory store to prove the Researcher's data is there.
    print(f"\n{'─' * 60}")
    print("MEMORY INSPECTION (after Phase 1)")
    print(f"{'─' * 60}")
    for key, value in memory_state.items():
        preview = value[:120].replace("\n", " ")
        print(f"  Key: '{key}' → {preview}...")
    print(f"  Total keys stored: {len(memory_state)}")

    # Phase 2: Writer runs and recalls from memory.
    await phase_two(topic)

    # Clean up the session.
    await shared_memory.clear("demo-session")
    print("\nSession 'demo-session' cleared.")


if __name__ == "__main__":
    asyncio.run(main())
