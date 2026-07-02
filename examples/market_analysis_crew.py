"""
Market Analysis Crew — agentflowkit parallel DAG demo.

5-agent diamond pipeline that analyzes a stock/asset from multiple angles
simultaneously, then synthesizes into a final investment recommendation.

DAG structure:
  Level 0 (parallel): news_analyst, technical_analyst, sentiment_analyst
  Level 1 (parallel): bull_case_agent, bear_case_agent
  Level 2 (single):   portfolio_strategist  ← structured Pydantic output

Usage:
    export GROQ_API_KEY=your_key      # free at console.groq.com
    python examples/market_analysis_crew.py NVDA
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from agentflow import LLM, Agent, InMemoryCache, Pipeline, PipelineLogger
from agentflow.types import Event
from pydantic import BaseModel, Field

# ─── Output Schema ─────────────────────────────────────────────────────────────

class InvestmentRecommendation(BaseModel):
    ticker: str
    action: str = Field(description="BUY, SELL, or HOLD")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 – 1.0")
    price_target_rationale: str
    key_risks: list[str]
    time_horizon: str = Field(description="short / medium / long term")


# ─── Level 0 Agents (run in parallel) ─────────────────────────────────────────

@Agent(name="news_analyst", role="Financial News Analyst")
async def news_analyst(task: str, context: dict) -> str:
    return (
        f"Analyze the latest news, earnings, and macroeconomic developments "
        f"affecting {task}. Summarize the 3 most impactful recent events and "
        f"their likely effect on the stock price. Be concise (200 words max)."
    )


@Agent(name="technical_analyst", role="Technical Analysis Specialist")
async def technical_analyst(task: str, context: dict) -> str:
    return (
        f"Perform a technical analysis of {task}. Cover: trend direction, "
        f"key support/resistance levels, RSI/MACD signals, and volume patterns. "
        f"Conclude with a short-term price outlook. Be concise (200 words max)."
    )


@Agent(name="sentiment_analyst", role="Market Sentiment Analyst")
async def sentiment_analyst(task: str, context: dict) -> str:
    return (
        f"Assess market sentiment for {task}. Cover: social media buzz, "
        f"analyst consensus (buy/hold/sell ratio), short interest, and "
        f"institutional flows. Rate overall sentiment: Bullish / Neutral / Bearish. "
        f"Be concise (200 words max)."
    )


# ─── Level 1 Agents (run in parallel, depend on Level 0) ──────────────────────

@Agent(name="bull_case_agent", role="Bull Case Analyst")
async def bull_case_agent(task: str, context: dict) -> str:
    news = context["news_analyst"]
    tech = context["technical_analyst"]
    sentiment = context["sentiment_analyst"]
    return (
        f"Based on this analysis of {task}, build the strongest possible BULL case:\n\n"
        f"News: {news[:300]}\n\nTechnical: {tech[:300]}\n\nSentiment: {sentiment[:300]}\n\n"
        f"Provide 3 specific reasons the stock will outperform. Include a realistic upside target."
    )


@Agent(name="bear_case_agent", role="Bear Case Analyst")
async def bear_case_agent(task: str, context: dict) -> str:
    news = context["news_analyst"]
    tech = context["technical_analyst"]
    sentiment = context["sentiment_analyst"]
    return (
        f"Based on this analysis of {task}, build the strongest possible BEAR case:\n\n"
        f"News: {news[:300]}\n\nTechnical: {tech[:300]}\n\nSentiment: {sentiment[:300]}\n\n"
        f"Provide 3 specific risks that could cause the stock to underperform. Include a downside target."
    )


# ─── Level 2 Agent (synthesizer, depends on Level 1) ──────────────────────────

@Agent(
    name="portfolio_strategist",
    role="Senior Portfolio Strategist",
    output_schema=InvestmentRecommendation,
)
async def portfolio_strategist(task: str, context: dict) -> str:
    bull = context["bull_case_agent"]
    bear = context["bear_case_agent"]
    return (
        f"You are making a final investment decision on {task}.\n\n"
        f"Bull case:\n{bull[:500]}\n\nBear case:\n{bear[:500]}\n\n"
        f"Synthesize both views and respond with a JSON object matching this schema exactly:\n"
        f"{InvestmentRecommendation.model_json_schema()}\n\n"
        f"Use ticker='{task}', action must be 'BUY', 'SELL', or 'HOLD', "
        f"confidence between 0.0 and 1.0."
    )


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(llm: LLM) -> Pipeline:
    pipe = Pipeline(llm=llm, retry_failed_agents=1)

    # Level 0 — all three run concurrently
    pipe.add(news_analyst, timeout=30.0)
    pipe.add(technical_analyst, timeout=30.0)
    pipe.add(sentiment_analyst, timeout=30.0)

    # Level 1 — both run concurrently after Level 0 completes
    pipe.add(
        bull_case_agent,
        depends_on=["news_analyst", "technical_analyst", "sentiment_analyst"],
        timeout=30.0,
    )
    pipe.add(
        bear_case_agent,
        depends_on=["news_analyst", "technical_analyst", "sentiment_analyst"],
        timeout=30.0,
    )

    # Level 2 — sequential synthesizer
    pipe.add(
        portfolio_strategist,
        depends_on=["bull_case_agent", "bear_case_agent"],
        timeout=30.0,
    )

    return pipe


# ─── Terminal Display ──────────────────────────────────────────────────────────

LEVEL_COLORS = ["\033[94m", "\033[96m", "\033[92m"]  # blue, cyan, green
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def handle_event(event: Event, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    level = event.data.get("level", 0)
    level_color = LEVEL_COLORS[min(level, len(LEVEL_COLORS) - 1)]

    if event.type == "agent_start":
        agent = event.agent.replace("_", " ").title()
        print(f"  {color('▶', level_color)} {BOLD}{agent}{RESET}  {DIM}(level {level}){RESET}")

    elif event.type == "agent_complete":
        agent = event.agent.replace("_", " ").title()
        tokens = event.data.get("tokens", 0)
        cached = " [cached]" if event.data.get("cached") else ""
        dur = event.data.get("duration", 0)
        print(f"  {color('✓', '\033[92m')} {BOLD}{agent}{RESET}  "
              f"{DIM}{tokens} tokens · {dur:.1f}s{cached}{RESET}")

    elif event.type == "agent_error":
        print(f"  {color('✗', '\033[91m')} {event.agent}: {event.data.get('error', '')}")

    elif event.type == "pipeline_complete":
        total_tokens = event.data.get("total_tokens", 0)
        total_dur = event.data.get("total_duration", 0)
        levels = event.data.get("levels_executed", 0)
        wall = elapsed
        print(f"\n  {color('→', '\033[93m')} {BOLD}Pipeline complete{RESET}  "
              f"{DIM}{total_tokens} tokens · {levels} levels · "
              f"wall time {wall:.2f}s (LLM sum {total_dur:.2f}s){RESET}\n")

    elif event.type == "pipeline_error":
        print(f"\n  {color('✗ Pipeline error:', '\033[91m')} {event.data.get('error', '')}\n")


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main(ticker: str) -> None:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    use_groq = bool(os.environ.get("GROQ_API_KEY"))
    llm = LLM(
        model="llama-3.3-70b-versatile" if use_groq else "gpt-4o-mini",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1" if use_groq else None,
        cache=InMemoryCache(default_ttl=1800),
        max_tokens=1024,
    )

    pipe = build_pipeline(llm)
    log = PipelineLogger("market-analysis")

    print(f"\n{BOLD}Market Analysis Crew{RESET} — {color(ticker.upper(), BOLD)}")
    print(f"{DIM}Provider: {'Groq' if use_groq else 'OpenAI'} · "
          f"Model: {llm.model} · 5 agents · 3 parallel levels{RESET}\n")

    start = time.perf_counter()
    log.log_pipeline_start(ticker, agent_count=5, level_count=3)

    result = None
    async for event in pipe.stream(ticker.upper()):
        handle_event(event, start)
        if event.type == "pipeline_complete":
            # Fetch full result via run() for structured output access
            # (stream doesn't return PipelineResult directly)
            pass

    # Re-run with cache to get PipelineResult (all hits, essentially free)
    result = await pipe.run(ticker.upper())
    log.log_pipeline_complete(result.run_id, result.total_tokens, result.total_duration)

    # Print recommendation
    strategist = result.get("portfolio_strategist")
    if strategist and "validated_output" in strategist.metadata:
        rec = InvestmentRecommendation(**strategist.metadata["validated_output"])
        action_color = {
            "BUY": "\033[92m",
            "SELL": "\033[91m",
            "HOLD": "\033[93m",
        }.get(rec.action, "")

        print(f"{BOLD}Investment Recommendation{RESET}")
        print(f"  Action:     {color(BOLD + rec.action + RESET, action_color)}")
        print(f"  Confidence: {rec.confidence:.0%}")
        print(f"  Horizon:    {rec.time_horizon}")
        print(f"  Rationale:  {rec.price_target_rationale[:200]}")
        print("  Key risks:")
        for risk in rec.key_risks[:3]:
            print(f"    • {risk}")
    else:
        print("\nFinal output:")
        print(strategist.output if strategist else "(no output)")

    print(f"\n{DIM}Run ID: {result.run_id} | Cache hits: {result.agents_with_cache_hits}/5{RESET}\n")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    asyncio.run(main(ticker))
