# API Reference

Auto-generated from the source docstrings.

Everything on this page is covered by the stability contract in
[PUBLIC_API.md](https://github.com/KaramQ6/agentflow/blob/main/PUBLIC_API.md):
if a name is exported from the top-level `agentflow` package, it is documented
here. The opt-in modules at the bottom are explicitly *not* covered by semver.

## Core

::: agentflow.Agent

::: agentflow.AgentSpec

::: agentflow.BaseAgent

::: agentflow.Pipeline

::: agentflow.LLM

::: agentflow.SupervisorAgent

## Tools

::: agentflow.tool

::: agentflow.Tool

## Cost

::: agentflow.estimate_cost

::: agentflow.register_price

## Observability

::: agentflow.Hooks

::: agentflow.LoggingHooks

::: agentflow.PipelineLogger

::: agentflow.get_logger

## Caching & rate limiting

::: agentflow.ResponseCache

::: agentflow.InMemoryCache

::: agentflow.RedisCache

::: agentflow.RateLimiter

## Memory

::: agentflow.BaseMemory

::: agentflow.InMemoryContext

::: agentflow.RedisContext

::: agentflow.VectorContext

## Human-in-the-loop

::: agentflow.ApprovalPolicy

::: agentflow.PauseExecution

## Data models

::: agentflow.AgentResult

::: agentflow.PipelineResult

::: agentflow.LLMResponse

::: agentflow.Event

::: agentflow.EventEmitter

## Exceptions

::: agentflow.AgentFlowError

::: agentflow.AgentError

::: agentflow.AgentTimeoutError

::: agentflow.AgentOutputValidationError

::: agentflow.BudgetExceededError

::: agentflow.PipelineError

::: agentflow.LLMError

::: agentflow.ToolError

## Opt-in modules

Imported by full path, **not** covered by the stability contract. Each may
change or disappear in any release.

::: agentflow.contrib.otel.OTelHooks

::: agentflow.triggers.BaseTrigger

::: agentflow.triggers.MQTTTrigger

::: agentflow.triggers.TriggerPolicy

::: agentflow.triggers.PydanticTriggerPolicy

::: agentflow.triggers.MQTTDaemon

::: agentflow.swarm_routing.DynamicSupervisorAgent
