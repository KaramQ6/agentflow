"""Research crew example - multi-agent research pipeline.

Run: python examples/research_crew.py

Requires GROQ_API_KEY environment variable or replace the api_key below.
"""

import asyncio
import os

from agentflow import Agent, Pipeline, LLM


# Configure LLM (Groq free tier)
llm = LLM(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", ""),
)


@Agent(name="researcher", role="Research Analyst")
async def researcher(task: str, context: dict) -> str:
    return (
        f"Research the following topic and provide a comprehensive analysis "
        f"with key findings, statistics, and expert opinions:\n\n{task}"
    )


@Agent(name="writer", role="Content Writer")
async def writer(task: str, context: dict) -> str:
    research = context["researcher"]
    return (
        f"Based on the following research, write a well-structured article "
        f"with an introduction, 3-4 main sections, and a conclusion.\n\n"
        f"Topic: {task}\n\nResearch:\n{research}"
    )


@Agent(name="editor", role="Editor and Fact-Checker")
async def editor(task: str, context: dict) -> str:
    article = context["writer"]
    return (
        f"Review and improve the following article. Fix any issues with "
        f"clarity, structure, grammar, and factual accuracy. Return the "
        f"final polished version.\n\nArticle:\n{article}"
    )


async def main():
    topic = "The Impact of Large Language Models on Software Development in 2025"

    # Build pipeline
    pipe = Pipeline(llm=llm)
    pipe.add(researcher)
    pipe.add(writer, depends_on=["researcher"])
    pipe.add(editor, depends_on=["writer"])

    print(f"Running research crew on: {topic}\n")
    print("=" * 60)

    # Stream events
    async for event in pipe.stream(topic):
        if event.type == "agent_start":
            print(f"\n>> {event.agent} started...")
        elif event.type == "agent_complete":
            tokens = event.data.get("tokens", 0)
            duration = event.data.get("duration", 0)
            print(f"   {event.agent} done ({tokens} tokens, {duration:.1f}s)")
        elif event.type == "pipeline_complete":
            print(f"\n{'=' * 60}")
            print(f"Pipeline complete!")
            print(f"  Total tokens: {event.data.get('total_tokens', 0)}")
            print(f"  Total time: {event.data.get('total_duration', 0):.1f}s")

    # Also run without streaming to get the result
    result = await pipe.run(topic)
    print(f"\n{'=' * 60}")
    print("FINAL ARTICLE:")
    print("=" * 60)
    print(result.output[:2000])  # Print first 2000 chars


if __name__ == "__main__":
    asyncio.run(main())
