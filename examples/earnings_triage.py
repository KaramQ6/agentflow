"""Earnings-call triage — the agentflow showcase.

A realistic diamond DAG with genuine dependencies and parallelism:

                        ┌─ financials_analyst ──┐
    transcript_fetcher ─┼─ sentiment_analyst   ─┼─ risk_synthesizer ─ brief_writer
                        └─ competitor_scanner ──┘

- ``transcript_fetcher`` uses **tools** (ReAct loop) to pull the transcript.
- The three analysts run **in parallel** (watch the event stream).
- ``risk_synthesizer`` enforces a **typed output schema** (Pydantic).
- The run prints **wall time vs. summed agent time** and **per-agent USD cost**.
- The second run reuses an **LLM response cache** — same pipeline, ~zero cost.

Works with any OpenAI-compatible endpoint. Zero-key options:

    # Ollama (local):
    export AGENTFLOW_BASE_URL=http://localhost:11434/v1
    export AGENTFLOW_MODEL=llama3.2
    export AGENTFLOW_API_KEY=ollama

    # Groq (free tier at console.groq.com):
    export AGENTFLOW_BASE_URL=https://api.groq.com/openai/v1
    export AGENTFLOW_MODEL=llama-3.3-70b-versatile
    export AGENTFLOW_API_KEY=gsk_...

Run:  python examples/earnings_triage.py
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel

from agentflow import LLM, Agent, InMemoryCache, Pipeline, tool

# ── Tools for the fetcher (stand-ins for a real transcript API) ───────────────

FAKE_TRANSCRIPT = (
    "Q3 revenue grew 18% YoY to $2.4B, beating guidance of $2.2B. Gross margin "
    "compressed 120bps to 61% on cloud infrastructure costs. Management raised "
    "FY guidance but flagged elongating enterprise sales cycles and FX headwinds "
    "in EMEA. R&D spend up 31% driven by the new AI product line; CFO noted "
    "hiring freeze outside of engineering. Buyback expanded by $500M."
)


@tool
def fetch_transcript(ticker: str, quarter: str) -> str:
    """Fetch the earnings-call transcript for a ticker and quarter."""
    return f"[{ticker} {quarter} transcript]\n{FAKE_TRANSCRIPT}"


@tool
def fetch_consensus(ticker: str) -> dict:
    """Fetch analyst consensus estimates for a ticker."""
    return {"ticker": ticker, "revenue_est": "2.2B", "eps_est": 1.42}


# ── Agents ────────────────────────────────────────────────────────────────────


@Agent(name="transcript_fetcher", role="Financial Data Engineer",
       tools=[fetch_transcript, fetch_consensus])
async def transcript_fetcher(task: str, context: dict) -> str:
    return (
        f"Fetch the earnings transcript and consensus estimates for: {task}. "
        "Return the full transcript followed by how results compare to consensus."
    )


@Agent(name="financials_analyst", role="Equity Research Analyst")
async def financials_analyst(task: str, context: dict) -> str:
    return (
        "Extract the hard numbers (revenue, margins, guidance, buybacks) and "
        f"rate the quarter. Transcript package:\n{context['transcript_fetcher']}"
    )


@Agent(name="sentiment_analyst", role="Communications Analyst")
async def sentiment_analyst(task: str, context: dict) -> str:
    return (
        "Assess management tone: confidence, hedging language, what they "
        f"avoided saying. Transcript package:\n{context['transcript_fetcher']}"
    )


@Agent(name="competitor_scanner", role="Competitive Intelligence Analyst")
async def competitor_scanner(task: str, context: dict) -> str:
    return (
        "Identify competitive signals: pricing pressure, market-share hints, "
        f"category risks. Transcript package:\n{context['transcript_fetcher']}"
    )


class RiskAssessment(BaseModel):
    overall_rating: str  # "bullish" | "neutral" | "bearish"
    key_risks: list[str]
    key_positives: list[str]
    confidence: float


@Agent(name="risk_synthesizer", role="Portfolio Risk Manager",
       output_schema=RiskAssessment)
async def risk_synthesizer(task: str, context: dict) -> str:
    return (
        "Synthesize the three analyses below into a risk assessment. Respond "
        f"ONLY with JSON matching this schema:\n{RiskAssessment.model_json_schema()}\n\n"
        f"Financials: {context['financials_analyst']}\n\n"
        f"Sentiment: {context['sentiment_analyst']}\n\n"
        f"Competitive: {context['competitor_scanner']}"
    )


@Agent(name="brief_writer", role="Investment Committee Writer")
async def brief_writer(task: str, context: dict) -> str:
    # risk_synthesizer declared an output_schema, so its context value is the
    # validated dict — typed data, not a raw string.
    risk = context["risk_synthesizer"]
    return (
        f"Write a 5-sentence investment-committee brief for {task}. "
        f"Rating: {risk['overall_rating']} (confidence {risk['confidence']}). "
        f"Risks: {risk['key_risks']}. Positives: {risk['key_positives']}."
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────


def build_pipeline(llm: LLM) -> Pipeline:
    pipe = Pipeline(llm=llm, budget_usd=0.25)  # hard cost ceiling per run
    pipe.add(transcript_fetcher)
    pipe.add(financials_analyst, depends_on=["transcript_fetcher"])
    pipe.add(sentiment_analyst, depends_on=["transcript_fetcher"])
    pipe.add(competitor_scanner, depends_on=["transcript_fetcher"])
    pipe.add(
        risk_synthesizer,
        depends_on=["financials_analyst", "sentiment_analyst", "competitor_scanner"],
    )
    pipe.add(brief_writer, depends_on=["risk_synthesizer"])
    return pipe


async def run_once(pipe: Pipeline, task: str, label: str) -> None:
    print(f"\n━━━ {label} ━━━")
    async for event in pipe.stream(task):
        match event.type:
            case "agent_start":
                print(f"  ▶ {event.agent}  (level {event.data['level']})")
            case "agent_complete":
                cached = " [cache hit]" if event.data["cached"] else ""
                print(f"  ✓ {event.agent}  {event.data['tokens']} tok{cached}")
            case "pipeline_complete":
                d = event.data
                print(
                    f"\n  wall time: {d['wall_time']}s  "
                    f"(agent time summed: {d['total_duration']}s — parallelism won "
                    f"{round(d['total_duration'] - d['wall_time'], 1)}s back)"
                )
                print(f"  total cost: ${d['total_cost']:.6f}")
            case "pipeline_error":
                print(f"  ✗ pipeline error: {event.data['error']}")


async def main() -> None:
    llm = LLM(
        model=os.environ.get("AGENTFLOW_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("AGENTFLOW_BASE_URL"),
        api_key=os.environ.get("AGENTFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        cache=InMemoryCache(default_ttl=3600),
        temperature=0.2,
    )
    pipe = build_pipeline(llm)
    task = "ACME Corp Q3 2026 earnings call"

    await run_once(pipe, task, "Run 1 — cold (real LLM calls)")
    await run_once(pipe, task, "Run 2 — warm (response cache)")


if __name__ == "__main__":
    asyncio.run(main())
