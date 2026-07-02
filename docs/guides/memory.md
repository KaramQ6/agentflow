# Memory & Context

Agents in agentflow can share state across pipeline executions through a
pluggable memory layer. This lets you build persistent, cross-session
workflows where agents recall prior outputs without passing context
manually.

## How it works

Every agent that has a memory backend attached will, on each execution:

1. **Load** all prior outputs for its session from the store.
2. **Inject** them into the system prompt as a `[Memory]` block (each entry
   is truncated to 300 characters).
3. **Save** its own output back to the store after completion.

The `Pipeline` propagates a `session_id` to every agent before execution,
so all agents in the same run automatically target the same session
namespace.

## Quick start

Pass an `InMemoryContext` instance to the pipeline (or individual agents)
and set a `session_id`:

```python
from agentflow import LLM, Agent, Pipeline, InMemoryContext

# Create a shared memory store
memory = InMemoryContext(default_ttl=3600, max_entries=500)

llm = LLM(model="gpt-4o-mini", api_key="...")

@Agent(name="researcher", role="Research Analyst")
async def researcher(task: str, context: dict) -> str:
    return f"Research: {task}"

@Agent(name="writer", role="Content Writer")
async def writer(task: str, context: dict) -> str:
    return f"Write about: {task}"

# Phase 1: Researcher saves findings to memory
pipe1 = Pipeline(llm=llm, memory=memory, session_id="blog-42")
pipe1.add(researcher)
await pipe1.run("Edge AI in 2025")

# Phase 2: Writer recalls Researcher's findings from memory
pipe2 = Pipeline(llm=llm, memory=memory, session_id="blog-42")
pipe2.add(writer)
await pipe2.run("Edge AI in 2025")
```

The Writer's system prompt will automatically include the Researcher's
output from Phase 1, even though the two agents run in separate pipelines.

!!! tip "Session ID is the bridge"
    Pipelines that share a `session_id` and a `BaseMemory` backend will
    see each other's stored context. Different session IDs are fully
    isolated — use them to partition knowledge.

## InMemoryContext

`InMemoryContext` is a lightweight, in-process implementation backed by a
Python dictionary. It requires no external dependencies and is ideal for
development, testing, and single-process deployments.

```python
from agentflow import InMemoryContext

memory = InMemoryContext(
    default_ttl=3600,     # entries expire after 1 hour (default)
    max_entries=1000,      # LRU eviction per session (default; 0 = unlimited)
)
```

### TTL (Time-To-Live)

Every stored entry carries an expiration timestamp. When entries are loaded
via `load_context()`, stale entries are purged automatically. This prevents
memory bloat in long-running processes.

```python
# Entries live for 5 minutes
memory = InMemoryContext(default_ttl=300)
```

!!! info "Monotonic clock"
    TTL is measured against `time.monotonic()`, so system clock changes
    won't cause premature expiry.

### LRU Eviction

When a session exceeds `max_entries`, the least-recently-used entry is
evicted to make room. This keeps memory bounded without manual cleanup.

```python
# Allow at most 50 entries per session
memory = InMemoryContext(max_entries=50)

# When the 51st entry is saved under a session, the oldest entry is dropped.
```

### Thread safety

All `InMemoryContext` operations are guarded by an `asyncio.Lock`, making
them safe for concurrent use within a single event loop. For multi-process
or distributed setups, use `RedisContext` instead.

## Long-term memory backends

### RedisContext

Backed by Redis, suitable for multi-process and distributed deployments
where memory must survive process restarts.

```python
from agentflow import RedisContext

memory = RedisContext(
    url="redis://localhost:6379/0",
    prefix="agentflow:mem:",  # key namespace
    ttl=86400,                 # optional: expire entire session hash after 24h
)
```

Data is stored as a Redis hash per session, with all values JSON-serialised.
An optional global `ttl` parameter expires the entire hash after every write.

!!! warning "Requires redis package"
    Install the extra: `pip install agentflowkit[redis]`

### VectorContext

Powered by ChromaDB, this backend stores context as vector embeddings,
enabling semantic search over historical agent outputs.

```python
from agentflow import VectorContext

# In-memory ChromaDB (ephemeral)
memory = VectorContext(collection_name="agent_history")

# Persistent storage on disk
memory = VectorContext(
    collection_name="agent_history",
    persist_dir="./chroma_data",
)

# With a custom embedding function
from chromadb.utils import embedding_functions

ef = embedding_functions.OpenAIEmbeddingFunction(api_key="...")
memory = VectorContext(embedding_fn=ef)
```

Call `search_context()` to find semantically relevant past outputs:

```python
results = await memory.search_context("edge computing latency", top_k=3)
for r in results:
    print(r["document"], r["distance"])
```

!!! warning "Requires chromadb package"
    Install the extra: `pip install agentflowkit[redis]`

## BaseMemory interface

All memory backends implement the same abstract interface. You can swap
implementations without changing agent code:

```python
from agentflow import BaseMemory

class BaseMemory(ABC):
    async def save_context(self, session_id: str, key: str, value: Any) -> None: ...
    async def load_context(self, session_id: str) -> dict[str, Any]: ...
    async def clear(self, session_id: str) -> None: ...
    async def delete_key(self, session_id: str, key: str) -> None: ...
```

To implement a custom backend (e.g., Postgres, S3), subclass `BaseMemory`
and implement these four methods. Pass your instance to the pipeline and it
will be used automatically.

## Example: multi-agent shared memory

The file `examples/memory_chat_agents.py` demonstrates a complete workflow
where a Researcher and Writer share context across two pipeline runs:

```bash
python examples/memory_chat_agents.py
```

See the full API in the [reference](../reference.md).
