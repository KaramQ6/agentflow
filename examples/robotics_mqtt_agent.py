"""
Robotics MQTT Agent — event-driven pipeline triggered by sensor data.

Demonstrates the Pipeline.serve() daemon mode with an MQTTTrigger.  A two-agent
pipeline listens to factory sensor measurements, flags anomalies, and logs
diagnostics — all triggered autonomously by incoming MQTT messages.

DAG structure:
  Level 0: sensor_analyzer   — inspects sensor payload for anomalies
  Level 1: diagnostic_agent  — runs only when an anomaly is detected

Prerequisites:
    pip install agentflowkit[mqtt]

Usage:
    # Start a local MQTT broker (e.g. mosquitto) then run:
    python examples/robotics_mqtt_agent.py

    # Publish test messages from another terminal:
    mosquitto_pub -t "factory/arm1/sensors" -m '{"temp": 82, "vibration": 0.3}'
    mosquitto_pub -t "factory/arm1/sensors" -m '{"temp": 105, "vibration": 4.7}'
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentflow import LLM, Agent, MQTTTrigger, Pipeline, PipelineLogger
from agentflow.types import PipelineResult

# ─── Agents ──────────────────────────────────────────────────────────────────────


@Agent(name="sensor_analyzer", role="IoT Sensor Analyst", timeout=30)
async def sensor_analyzer(task: str, context: dict) -> str:
    return (
        f"You are an industrial IoT sensor analyst.  Examine the following "
        f"sensor reading and determine if any value indicates an anomaly "
        f"(threshold breach, sudden spike, sensor degradation, etc.).\n\n"
        f"SENSOR DATA:\n{task}\n\n"
        f"Respond with a concise assessment.  Start with 'ANOMALY DETECTED:' "
        f"if there is a fault, or 'NOMINAL:' if everything is within range."
    )


@Agent(name="diagnostic_agent", role="Robotics Diagnostics Specialist", timeout=30)
async def diagnostic_agent(task: str, context: dict) -> str:
    analyzer_output = context.get("sensor_analyzer", "")
    return (
        f"The sensor analyzer flagged a potential issue.  Based on the "
        f"original sensor data below and the analyzer's assessment, produce "
        f"a root-cause hypothesis and recommend a corrective action.\n\n"
        f"ANALYZER ASSESSMENT:\n{analyzer_output}\n\n"
        f"ORIGINAL DATA:\n{task}\n\n"
        f"Output format: 'ROOT CAUSE: ...' then 'ACTION: ...'"
    )


# ─── Conditional gate ────────────────────────────────────────────────────────────


def anomaly_detected(context: dict[str, str]) -> bool:
    """Only run diagnostics when the sensor analyzer finds an anomaly."""
    output = context.get("sensor_analyzer", "")
    return output.upper().startswith("ANOMALY")


# ─── Main ────────────────────────────────────────────────────────────────────────


async def main():
    llm = LLM(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        provider="groq",
    )

    pipe = Pipeline(
        llm=llm,
        hooks=PipelineLogger(verbose=True),
    )
    pipe.add(sensor_analyzer)
    pipe.add(diagnostic_agent, depends_on=["sensor_analyzer"], condition=anomaly_detected)

    trigger = MQTTTrigger(
        broker=os.environ.get("MQTT_BROKER", "localhost"),
        port=int(os.environ.get("MQTT_PORT", "1883")),
        topic=os.environ.get("MQTT_TOPIC", "factory/+/sensors"),
        prompt_template="Analyze this sensor data: {data}",
    )

    def on_done(result: PipelineResult) -> None:
        print(f"\n{'='*60}")
        print(f"Run {result.run_id} completed | {result.total_tokens} tokens | "
              f"{result.total_duration:.2f}s")
        for name, ar in result.results.items():
            print(f"  [{name}] {ar.output[:120]}...")
        print(f"{'='*60}")

    print(f"Listening on MQTT broker {trigger._broker}:{trigger._port} [{trigger._topic}]")
    print("Publish a JSON payload to trigger the pipeline. Press Ctrl+C to stop.\n")

    await pipe.serve(
        trigger,
        max_concurrent=3,
        on_result=on_done,
    )


if __name__ == "__main__":
    asyncio.run(main())
