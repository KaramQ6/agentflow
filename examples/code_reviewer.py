"""Code reviewer example - AI code review pipeline.

Run: python examples/code_reviewer.py

Requires GROQ_API_KEY environment variable.
"""

import asyncio
import os

from agentflow import Agent, Pipeline, LLM


llm = LLM(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", ""),
)


@Agent(name="analyzer", role="Code Analyzer")
async def analyzer(task: str, context: dict) -> str:
    return (
        f"Analyze the following code for potential issues including bugs, "
        f"security vulnerabilities, performance problems, and code smells. "
        f"List each issue with its severity (critical/warning/info).\n\n"
        f"Code:\n```\n{task}\n```"
    )


@Agent(name="suggester", role="Code Improvement Specialist")
async def suggester(task: str, context: dict) -> str:
    analysis = context["analyzer"]
    return (
        f"Based on the following code analysis, provide specific code "
        f"improvements. For each issue found, show the exact fix with "
        f"before/after code snippets.\n\n"
        f"Original code:\n```\n{task}\n```\n\n"
        f"Analysis:\n{analysis}"
    )


SAMPLE_CODE = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    data = result.fetchone()
    password = data['password']
    return {"id": data['id'], "name": data['name'], "password": password}
'''


async def main():
    pipe = Pipeline(llm=llm)
    pipe.add(analyzer)
    pipe.add(suggester, depends_on=["analyzer"])

    print("Code Review Pipeline")
    print("=" * 60)

    async for event in pipe.stream(SAMPLE_CODE.strip()):
        if event.type == "agent_start":
            print(f"\n>> {event.agent} reviewing...")
        elif event.type == "agent_complete":
            print(f"   Done ({event.data.get('tokens', 0)} tokens)")
        elif event.type == "pipeline_complete":
            print(f"\nReview complete in {event.data.get('total_duration', 0):.1f}s")

    result = await pipe.run(SAMPLE_CODE.strip())
    print(f"\n{'=' * 60}")
    print("REVIEW RESULT:")
    print("=" * 60)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
