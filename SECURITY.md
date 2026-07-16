# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.6.x | ✅ |
| < 0.6 | ❌ — upgrade; 0.5.0's only wheel was a Windows/cp311 binary |

## Reporting a vulnerability

Please report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/KaramQ6/agentflow/security/advisories/new)
— do **not** open a public issue. You can expect an acknowledgement within a
few days. Coordinated disclosure after a fix is released is appreciated.

## Scope notes

- **`agentflow.sandbox` is a code-execution surface.** It runs model-generated
  code in Docker or a subprocess. It is opt-in (never enabled implicitly),
  imported only by full path, and explicitly **outside the semver stability
  contract** (see [PUBLIC_API.md](PUBLIC_API.md)). Review it yourself before
  using it in production, treat the subprocess backend as *not* a security
  boundary, and prefer the Docker backend with a locked-down image.
- **Prompt injection is not a solved problem.** Tool-calling agents execute
  the tools you give them with arguments chosen by the model. Gate anything
  irreversible behind `ApprovalPolicy` (human-in-the-loop) and validate tool
  arguments at the tool boundary.
- **Secrets**: agentflow never logs API keys; structured logs include prompts
  and outputs — scrub upstream if your prompts contain sensitive data.
