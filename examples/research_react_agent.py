"""Research ReAct Agent — tool-calling agent with web search and structured output.

Demonstrates a ReAct agent that searches for information about a topic using
multiple search queries, synthesises the findings, and returns a structured
JSON summary validated against a Pydantic schema.

Run: python examples/research_react_agent.py "quantum computing"
Requires GROQ_API_KEY (free at console.groq.com) or edit the api_key below.
"""

from __future__ import annotations

import asyncio
import json
import os

from pydantic import BaseModel, Field

from agentflow import LLM, Agent, Pipeline, tool

# ─── Mock Search Tool ──────────────────────────────────────────────────────────

# A simulated search endpoint. In production, replace with a real API
# (SerpAPI, Tavily, Brave, etc.) or @tool wrappers around httpx requests.
_MOCK_INDEX: dict[str, list[dict[str, str]]] = {
    "quantum computing": [
        {"title": "What Is Quantum Computing? | IBM", "snippet": "Quantum computing uses qubits that can exist in superposition, enabling exponential speedups for certain problems like factoring and simulation."},
        {"title": "Quantum Supremacy Explained | Nature", "snippet": "Google's Sycamore processor achieved quantum supremacy in 2019 by performing a specific computation in 200 seconds that would take a classical supercomputer 10,000 years."},
    ],
    "machine learning": [
        {"title": "Introduction to ML | Google AI", "snippet": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming, using algorithms like neural networks and decision trees."},
        {"title": "Deep Learning Revolution | MIT Tech Review", "snippet": "Deep learning, powered by multi-layer neural networks and massive datasets, has driven breakthroughs in computer vision, NLP, and generative AI since 2012."},
    ],
    "climate change": [
        {"title": "Climate Change Overview | NASA", "snippet": "Global temperatures have risen ~1.1°C since pre-industrial times due to greenhouse gas emissions, leading to more frequent extreme weather events and sea level rise of ~3.6 mm/year."},
        {"title": "Net Zero Roadmap 2050 | IEA", "snippet": "Achieving net-zero emissions by 2050 requires tripling renewable capacity by 2030, electrifying transport, and deploying carbon capture technologies at gigatonne scale."},
    ],
}


@tool
def search(query: str) -> str:
    """Search for information about a topic. Returns a list of relevant
    articles with titles and snippets. Provide a specific, targeted query."""
    results = _MOCK_INDEX.get(query.lower(), [])
    if not results:
        return json.dumps({"query": query, "results": [], "hint": "try a more specific query"})
    return json.dumps({"query": query, "count": len(results), "results": results})


# ─── Structured Output Schema ──────────────────────────────────────────────────

class ResearchSummary(BaseModel):
    topic: str = Field(description="The research topic")
    summary: str = Field(description="A concise synthesis of findings (2-4 sentences)")
    key_facts: list[str] = Field(description="3-5 key facts discovered during research")
    sources_used: list[str] = Field(description="Titles of sources referenced")
    confidence: str = Field(description="high / medium / low — based on source quality")


# ─── ReAct Agent ───────────────────────────────────────────────────────────────

@Agent(
    name="researcher",
    role="Research Analyst who uses the search tool to gather information "
          "and produces structured, evidence-based summaries",
    tools=[search],
    output_schema=ResearchSummary,
    max_tool_iterations=6,
)
async def researcher(task: str, context: dict) -> str:
    """Build a research prompt that instructs the model to search and synthesise.

    The model sees this as a user message and decides which tools to call
    (if any). Since ``search`` is attached, it runs a ReAct loop — the model
    may issue multiple search queries, observe results, then produce a final
    JSON answer matching ResearchSummary.
    """
    return (
        f"You are researching the following topic: **{task}**.\n\n"
        f"Follow these steps:\n"
        f"1. Use the `search` tool with a few well-chosen queries to gather information.\n"
        f"2. Synthesise what you found into a clear, concise research summary.\n"
        f"3. Return a JSON object matching this schema:\n"
        f"   {json.dumps(ResearchSummary.model_json_schema(), indent=2)}\n\n"
        f"Important: only use information from the search results. "
        f"Do not fabricate facts."
    )


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main(topic: str) -> None:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
        return

    use_groq = bool(os.environ.get("GROQ_API_KEY"))
    llm = LLM(
        model="llama-3.3-70b-versatile" if use_groq else "gpt-4o-mini",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1" if use_groq else None,
        max_tokens=1024,
    )

    pipe = Pipeline(llm=llm)
    pipe.add(researcher)

    print(f"\nResearching: {topic}\n{'=' * 60}")

    result = await pipe.run(topic)

    # ReAct trace — show every tool invocation the model made
    agent_result = result.get("researcher")
    if agent_result:
        trace = agent_result.metadata.get("tool_calls", [])
        if trace:
            print("\nTOOL CALLS MADE:")
            for call in trace:
                args_preview = str(call["arguments"])[:120]
                result_preview = str(call["result"])[:200]
                print(f"  ▶ {call['tool']}({args_preview})")
                print(f"    ← {result_preview}")

        # Structured output
        if "validated_output" in agent_result.metadata:
            rec = ResearchSummary(**agent_result.metadata["validated_output"])
            print(f"\n{'─' * 60}")
            print("RESEARCH SUMMARY")
            print(f"  Topic:      {rec.topic}")
            print(f"  Confidence: {rec.confidence.upper()}")
            print(f"  Sources:    {', '.join(rec.sources_used[:4])}")
            print(f"  Summary:    {rec.summary[:300]}")
            print("  Key Facts:")
            for i, fact in enumerate(rec.key_facts, 1):
                print(f"    {i}. {fact}")

        print(f"\nTokens: {result.total_tokens} | Cost: ${result.total_cost:.6f} | "
              f"Duration: {result.total_duration:.1f}s")


if __name__ == "__main__":
    topic = input("Topic: ").strip() or "quantum computing"
    asyncio.run(main(topic))
