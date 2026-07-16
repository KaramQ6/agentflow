"""
Drone Telemetry Agent — reactive drone safety monitor via MQTT.

Demonstrates the MQTTDaemon with a PydanticTriggerPolicy that spawns an
agent pipeline only when critical thresholds are breached (battery < 15%
or altitude drop rate > 5 m/s).  Incoming telemetry is strictly validated
against a Pydantic model, and pipeline execution is fully non-blocking.

DAG structure:
  Level 0: threat_assessor     — evaluates severity of the telemetry breach
  Level 1: emergency_responder — proposes corrective action if threat is real

Prerequisites:
    pip install agentflowkit[mqtt]

Usage:
    # Start a local MQTT broker (e.g. mosquitto) then run:
    python examples/drone_telemetry_agent.py

    # Simulate a critical battery warning:
    mosquitto_pub -t "drones/drone-01/telemetry" -m '{"battery": 12, "altitude": 80, "altitude_drop_rate": 2, "gps_lat": 37.7749, "gps_lon": -122.4194}'

    # Simulate rapid descent:
    mosquitto_pub -t "drones/drone-01/telemetry" -m '{"battery": 85, "altitude": 50, "altitude_drop_rate": 7.5, "gps_lat": 37.7749, "gps_lon": -122.4194}'

    # Normal telemetry (no pipeline triggered):
    mosquitto_pub -t "drones/drone-01/telemetry" -m '{"battery": 92, "altitude": 100, "altitude_drop_rate": 0.3, "gps_lat": 37.7749, "gps_lon": -122.4194}'
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field  # noqa: E402

from agentflow import LLM, Agent, Pipeline, PipelineLogger  # noqa: E402
from agentflow.events import MQTTDaemon, PydanticTriggerPolicy  # noqa: E402
from agentflow.types import PipelineResult  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Pydantic telemetry model ───────────────────────────────────────────────────


class DroneTelemetry(BaseModel):
    """Strictly validate incoming drone telemetry payloads."""

    battery: float = Field(..., ge=0, le=100, description="Battery percentage")
    altitude: float = Field(..., ge=0, description="Altitude in meters")
    altitude_drop_rate: float = Field(..., ge=0, description="Descent rate in m/s")
    gps_lat: float = Field(..., ge=-90, le=90)
    gps_lon: float = Field(..., ge=-180, le=180)


# ─── Trigger policy ─────────────────────────────────────────────────────────────


def critical_condition(data: DroneTelemetry) -> bool:
    """Trigger when battery is critically low or the drone is falling rapidly."""
    return data.battery < 15 or data.altitude_drop_rate > 5


telemetry_policy = PydanticTriggerPolicy(
    model=DroneTelemetry,
    condition=critical_condition,
    prompt_template=(
        "ALERT: Drone telemetry anomaly. Battery: {battery}% | "
        "Altitude: {altitude}m | Descent Rate: {altitude_drop_rate} m/s | "
        "GPS: ({gps_lat}, {gps_lon}). Assess severity and recommend action."
    ),
)


# ─── Agents ─────────────────────────────────────────────────────────────────────


@Agent(name="threat_assessor", role="Drone Safety Assessor", timeout=30)
async def threat_assessor(task: str, context: dict) -> str:
    return (
        f"You are a drone flight safety analyst.  Examine the telemetry alert "
        f"below and classify the severity as CRITICAL, WARNING, or NOMINAL. "
        f"Consider whether the condition warrants an emergency landing or "
        f"merely a routine warning.\n\n"
        f"TELEMETRY:\n{task}\n\n"
        f"Respond with: SEVERITY: <level>\nFINDINGS: <brief assessment>"
    )


@Agent(name="emergency_responder", role="Drone Emergency Responder", timeout=30)
async def emergency_responder(task: str, context: dict) -> str:
    assessment = context.get("threat_assessor", "")
    return (
        f"Based on the flight safety assessment below, determine the "
        f"appropriate emergency protocol.  Options include: immediate "
        f"return-to-home (RTH), controlled descent to nearest safe zone, "
        f"deploy parachute, or continue monitoring.\n\n"
        f"ASSESSMENT:\n{assessment}\n\n"
        f"ORIGINAL ALERT:\n{task}\n\n"
        f"Respond with: ACTION: <specific protocol>\nJUSTIFICATION: <reasoning>"
    )


# ─── Non-blocking pipeline handler ──────────────────────────────────────────────


async def handle_trigger(
    task_prompt: str,
    payload: dict,
    context: dict,
) -> None:
    """Spawned via ``asyncio.create_task`` — never blocks the MQTT listener."""
    logger = logging.getLogger("drone_telemetry")

    llm = LLM(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        provider="groq",
    )

    pipe = Pipeline(
        llm=llm,
        hooks=PipelineLogger(verbose=True),
    )
    pipe.add(threat_assessor)
    pipe.add(emergency_responder, depends_on=["threat_assessor"])

    try:
        result: PipelineResult = await pipe.run(task_prompt)
        logger.info(
            "Run %s: severity assessed in %.2fs (%d tokens, $%.4f)",
            result.run_id,
            result.total_duration,
            result.total_tokens,
            result.total_cost,
        )
        for name, ar in result.results.items():
            logger.info("  [%s] %s", name, ar.output[:200])
    except Exception as exc:
        logger.error("Pipeline failed for telemetry event: %s", exc)


# ─── Main ────────────────────────────────────────────────────────────────────────


async def main():
    broker = os.environ.get("MQTT_BROKER", "localhost")
    port = int(os.environ.get("MQTT_PORT", "1883"))
    topic = os.environ.get("MQTT_TOPIC", "drones/+/telemetry")

    daemon = MQTTDaemon(
        broker=broker,
        port=port,
        topic=topic,
        policy=telemetry_policy,
        handler=handle_trigger,
    )

    print(f"Drone telemetry monitor listening on {broker}:{port} [{topic}]")
    print("Publish JSON telemetry to trigger alerts. Press Ctrl+C to stop.\n")
    await daemon.serve()


if __name__ == "__main__":
    asyncio.run(main())
