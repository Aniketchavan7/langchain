---
type: Workflow
title: "Streaming: Token-by-Token Output and Events"
description: "Streaming in chat models delivers responses incrementally through chunks and structured events. The system supports V1 callback-based streaming, V2 event-based iteration, and V3 protocol events with typed projections for text, reasoning, tool calls, and usage metadata."
tags: [streaming, chat-models, events, protocol, chunks, callbacks, reasoning, tool-calls]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-20T08:17:03.730Z
sources:
  - id: openwiki-source-a9d15707c8dcabbe5cf98edb
    resource: repo://libs/core/langchain_core/language_models/_compat_bridge.py
  - id: openwiki-source-a702d4b2162ddc17e6bee5fd
    resource: repo://libs/core/langchain_core/language_models/_finish_reason.py
  - id: openwiki-source-5f8bc32563177d89fbab9b2f
    resource: repo://libs/core/langchain_core/language_models/chat_model_stream.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-5ca288f79e21018a1fb8c21e
    resource: repo://libs/core/tests/unit_tests/language_models/test_finish_reason.py
generated: { by: "openwiki/0.5.0", at: "2026-11-20T08:17:03.730Z" }
---

# Streaming: Token-by-Token Output and Events

LangChain chat models support multiple streaming modes that allow responses to arrive token-by-token rather than waiting for the full output. Each mode enables different use cases: V1 callback-based for observability, V2 raw event iteration, and V3 protocol events with structured accumulation and projections.

## Streaming Modes and Protocols

### V1 Callback-Based Streaming (Legacy)

The V1 mode uses the callback system (`on_llm_new_token`, `on_llm_end`, `on_llm_error`) to signal streaming progress. Tokens arrive through `on_llm_new_token`, and the final message is delivered via `on_llm_end`. This mode remains useful for observability but lacks structured event metadata.

### V2 Event-Based Iteration (Runnable.stream_events)

For `version="v1"` and `"v2"` (the default), `stream_events()` returns an `Iterator[StreamEvent]` following the standard Runnable event protocol. Event types include:
- `on_chat_model_start`: Model invocation begins
- `on_chat_model_stream`: Chunk received from the underlying provider
- `on_chat_model_end`: Model invocation complete

This mode is compatible with standard Runnable event dispatch and observability tools.

### V3 Protocol Events and Typed Projections

For `version="v3"` (beta), `stream_events()` returns a **`ChatModelStream`** (sync) or **`AsyncChatModelStream`** (async) object. This mode exposes:

- **Raw event iteration**: The stream object itself is iterable, yielding structured protocol events
- **Typed projections**: `.text`, `.reasoning`, `.tool_calls`, `.usage`, `.output` accumulate deltas and provide final values
- **Content-block protocol**: Messages are assembled incrementally from `message-start`, per-block `content-block-*` events, and `message-finish`

## Content-Block Protocol Lifecycle (V3)

The V3 event stream follows a structured lifecycle:

```
message-start
  -> content-block-start (index 0)
  -> content-block-delta (text chunk)
  -> content-block-delta (text chunk)
  -> content-block-finish (finalized text block)
  -> content-block-start (index 1, e.g. tool call)
  -> content-block-delta (tool args chunk)
  -> content-block-finish (finalized tool call)
-> message-finish (usage, finish reason metadata)
```

Each content block is identified by a **wire index** (a sequential uint assigned on first observation). Providers may emit blocks in any order (e.g., Anthropic interleaves parallel tool calls by their source-side index), and the bridge translates source indices to wire indices for stability.

### Event Structure

**`message-start`**: Signals the beginning of the response. Contains optional `metadata` (provider, model) and `id`.

**`content-block-start`**: Signals the beginning of a new content block at a given index. The `content` field holds a skeleton with minimal data:
- Text/reasoning blocks: empty text/reasoning field, other fields preserved
- Tool-call blocks: id, name, index preserved; args empty
- Self-contained types (image, file, etc.): heavy fields (data, output, etc.) omitted; correlation fields preserved

**`content-block-delta`**: Carries an incremental delta for a block. Delta types include:
- `text-delta`: new text chunk
- `reasoning-delta`: new reasoning chunk  
- `tool-call-delta`: new tool args (string slice, not JSON-parsed)
- `data-delta`: new data chunk (e.g., base64-encoded media)

**`content-block-finish`**: Finalizes a block at an index. The `content` field holds the complete, parsed block:
- Text/reasoning: fully accumulated text
- Tool-call blocks: args parsed to dict; invalid JSON becomes `invalid_tool_call` with the raw string
- Other types: as emitted

**`message-finish`**: Signals completion. Contains optional `usage` (input/output token counts with optional breakdown), `finish_reason` / `normalized_finish_reason` / `finish_state` from the provider, `metadata`, and `additional_kwargs` for provider-specific extensions.

## Reasoning Block Streaming (o1/o3)

Models with extended thinking (o1, o3) emit reasoning blocks that stream incrementally. The protocol treats reasoning like text:

- A `content-block-start` event opens the reasoning block
- Multiple `content-block-delta` events (type `reasoning-delta`) carry chunks of reasoning text
- A `content-block-finish` event closes the block with the full accumulated reasoning

The `.reasoning` projection on the stream accumulates and provides access to the full reasoning text without blocking.

## Compat Bridge: Converting Chunks to Protocol Events

The **compat bridge** (`_compat_bridge.py`) converts raw chunk streams from providers into standardized protocol events. Its design:

- **Single protocol view**: `AIMessageChunk.content_blocks` is the authoritative source of block structure, using three-tier lookup (output_version, registered translator, best-effort parsing)
- **Delta-slice semantics**: Each chunk carries only its own increment; the bridge emits deltas straight through without accumulating
- **Per-block accumulation**: State is accumulated per-index (by source-side `index` or positional fallback) only for finalization; live deltas bypass the accumulator
- **Block finalization**: Tool-call chunks are parsed on finish; invalid JSON becomes `invalid_tool_call` preserving the raw args

Key functions:
- **`chunks_to_events(chunks, message_id=None, generation_info=None)`**: Converts an iterator of `ChatGenerationChunk` to protocol events
- **`achunks_to_events(chunks, ...)`**: Async variant
- **`message_to_events(msg, message_id=None)`**: Replays a finalized `AIMessage` as synthetic events (for cache hits, checkpoints, graph node returns)
- **`finalize_tool_call_chunk(raw_args, id_, name, extras, finalized_type)`**: Parses accumulated tool-call args; shared with stream accumulators

## ChatModelStream: Typed Accumulation and Projections (V3)

**`ChatModelStream`** (sync) and **`AsyncChatModelStream`** (async) provide high-level accumulation and projection interfaces for V3 protocol events.

### State and Accumulators

The stream maintains:
- **Text accumulator** (`_text_acc`): Full accumulated output text
- **Per-block text** (`_text_per_block`): Text storage keyed by wire index (prevents cross-contamination)
- **Reasoning accumulator** (`_reasoning_acc`): Full accumulated reasoning
- **Tool-call store** (`_tool_call_chunks`, `_server_tool_call_chunks`): Per-index chunk stores for args assembly
- **Finalized blocks** (`_blocks`): Wire-indexed snapshot of every finalized block
- **Usage** (`_usage_value`): Summed token counts
- **Finish-reason fields**: `finish_reason`, `normalized_finish_reason`, `finish_state`

### Projections

Projections are accumulation objects that yield deltas and buffer final values:

- **`.text`**: `SyncTextProjection` or `AsyncTextProjection` — yields text deltas, final value is full output text
- **`.reasoning`**: Similar to `.text` — yields reasoning deltas, final value is accumulated reasoning
- **`.tool_calls`**: `SyncProjection` or `AsyncProjection` — yields `ToolCallChunk` deltas (with id, name, args slice), final value is list of finalized `ToolCall` dicts
- **`.usage`**: `SyncProjection` or `AsyncProjection` — yields full `UsageInfo` on each update, final value is accumulated usage
- **`.output`**: `SyncProjection` or `AsyncProjection` — yields `AIMessage` snapshots as blocks finalize, final value is complete assembled `AIMessage`

Projections support multiple independent consumers via replay semantics — each call to `__iter__` or `__aiter__` starts at position 0, replaying all accumulated deltas.

### Event Dispatch and Accumulation

The `dispatch(event)` method routes protocol events:

- **`message-start`**: Records start metadata
- **`content-block-start`**: Seeds chunk stores for tool-call blocks (seeds `id`, `name`, `index`)
- **`content-block-delta`**: 
  - Text/reasoning deltas: accumulate per-block and globally, push to projection
  - Tool-call deltas: append args slice to store, push `ToolCallChunk` with id/name from store
  - Legacy block-delta format: convert and re-route
- **`content-block-finish`**: Parse tool args to dict or `invalid_tool_call`, append to finalized blocks, complete output snapshot
- **`message-finish`**: Record usage, finish reason, finalize projections, assemble final message

### Output Assembly

The stream constructs `AIMessage` with:
- `content`: List of finalized blocks (text, reasoning, tool calls, images, etc.)
- `usage_metadata`: Accumulated token counts
- `response_metadata`: Provider metadata including finish reason fields
- `additional_kwargs`: Provider-specific fields (e.g., Gemini thought signatures)

## Streaming Usage and Examples

### Basic Streaming (V1 Callback, Default)

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

model.invoke(
    "What is 2+2?",
    config={"callbacks": [StreamingStdOutCallbackHandler()]},
)
# Prints tokens as they arrive
```

### V2 Event Streaming

```python
for event in model.stream_events(
    "What is 2+2?",
    version="v2"  # default
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].message.content)
```

### V3 Protocol Streaming with Typed Projections (Beta)

```python
stream = model.stream_events(
    "Explain quantum computing",
    version="v3"
)

# Iterate raw protocol events
for event in stream:
    if event["event"] == "content-block-delta":
        print(event["delta"])

# Access typed projections
print("Text output:", stream.text)        # Drain and get full text
print("Reasoning:", stream.reasoning)      # Full reasoning blocks
print("Tool calls:", stream.tool_calls)    # Finalized tool calls list

# Wait for complete message
message = await stream                      # AsyncChatModelStream: await for final AIMessage
# or
message = stream.output.get()               # Sync: drain for final AIMessage
```

### Streaming Tool Calls

Tool calls arrive incrementally as `tool-call-delta` events:

```python
stream = model.stream_events(
    "Call the calculator with 2+2",
    version="v3"
)

for call_chunk in stream.tool_calls:
    # call_chunk is {'type': 'tool_call_chunk', 'id': '...', 'name': '...', 'args': '...piece...'}
    print(f"Partial args for {call_chunk['name']}: {call_chunk['args']}")

final_calls = stream.tool_calls.get()  # Finalized ToolCall dicts with parsed args dicts
for call in final_calls:
    print(f"Call {call['id']}: {call['name']}({call['args']})")
```

### Streaming with Reasoning (o1/o3)

```python
stream = model.stream_events(
    "Prove Fermat's Last Theorem",
    version="v3"
)

# Consume reasoning as it arrives
async for reasoning_chunk in stream.reasoning:
    print(reasoning_chunk, end="", flush=True)

# Or drain for final values
print("Final reasoning:", stream.reasoning)
print("Final answer:", stream.text)
```

### Accessing Usage Metadata

Token counts are available during streaming:

```python
stream = model.stream_events(
    "Hello",
    version="v3"
)

# Consume some events
for _ in range(5):
    next(iter(stream))

# Check usage so far
usage = stream.usage.get()
print(f"Output tokens so far: {usage.get('output_tokens', 0)}")

# Final usage after all events
for event in stream:
    pass  # Drain stream

final_usage = stream.usage.get()
print(f"Final: {final_usage['input_tokens']} in, {final_usage['output_tokens']} out")
```

### Handling Streaming Errors

```python
try:
    stream = model.stream_events(
        "Query with error",
        version="v3"
    )
    for event in stream:
        pass
except Exception as e:
    # Errors during streaming are raised when consuming events
    print(f"Stream error: {e}")
    # Check partial accumulation
    if stream.has_events:
        print(f"Partial text: {stream.text}")
```

## Streaming Lifecycle and State Management

### Synchronous Streaming (ChatModelStream)

1. **Lazy start**: The stream does not begin pulling chunks until the first consumer accesses it (`__iter__`, `__aiter__`, or `.get()`)
2. **Pump callback**: The stream driver (`_chat_model_stream_v3`) installs a `pump_one()` callback that pulls one event and dispatches it to the stream
3. **Callback integration**: After `ensure_started()`, `on_chat_model_start` is called with the run_id; after each event, checks are made for stream completion and `on_llm_end` or `on_llm_error`
4. **Backpressure**: Consumers pull via the projection iterators or `.get()` drain calls, which invoke the pump until the stream is exhausted

### Asynchronous Streaming (AsyncChatModelStream)

1. **Task-based production**: A background `asyncio.Task` runs `_produce()`, which consumes from `_aiter_v2_events` and dispatches to the stream
2. **Event notification**: `asyncio.Event` signals waiters when new deltas arrive or the final value is set
3. **Multi-consumer support**: Each projection iterator maintains its own cursor and re-checks the shared event on wake
4. **Cancellation handling**: If the task is cancelled before `_produce` starts, `_on_aclose_fail` fires to close the callback lifecycle

### Finish Reason Handling

The `_finish_fields()` function normalizes provider finish reasons:

- Passes through `finish_reason` (raw provider value)
- Passes through `normalized_finish_reason` from the provider
- Applies two overrides:
  - If a `refusal` content block is present, normalized reason becomes `"refusal"`
  - If `normalized_finish_reason == "stop"` and tool calls are present, becomes `"tool_call"`
- Passes through `finish_state` (e.g., `"suspended"` for pause_turn)

This ensures that finish-reason semantics are preserved end-to-end through streaming.

## Integration with Callbacks and Observability

Streaming is fully integrated with the callback system:

- **`on_chat_model_start`** fires when the stream is first consumed, using the run_id as the message identifier
- **`on_llm_new_token`** (V1 only): Not used in V3; use projections instead
- **`on_chat_model_stream`** (V2): Emitted for each raw event
- **`on_llm_end`** fires after the message is fully assembled and final
- **`on_llm_error`** fires if streaming raises an exception

This allows streaming to integrate with LangSmith, logging, and custom callback handlers seamlessly.

## Streaming Configuration

### Output Version

The model's `output_version` field affects how content is structured on non-streaming paths (`invoke()`, `astream()`), but V3 streaming **always produces v1-shaped content** (list of content blocks) in the assembled `AIMessage`.

### Disabling Streaming

The `disable_streaming` field can be set to:
- `False` (default): Use streaming when available
- `True`: Always use `invoke()`
- `"tool_calling"`: Use `invoke()` only when tools are bound

This allows swapping models with incompatible streaming implementations.

## Extension Points

Custom chat model implementations can:

1. **Override `_stream(messages, stop, run_manager, **kwargs)`** to provide native streaming chunks
2. **Emit `AIMessageChunk` with `content_blocks`** (via `content`, `tool_call_chunks`, or custom translators) so the bridge can convert to protocol events
3. **Carry finish-reason metadata** in `response_metadata` with keys `finish_reason`, `normalized_finish_reason`, `finish_state` so they reach the protocol and assembled message
4. **Preserve provider-specific metadata** in `additional_kwargs` (e.g., Gemini thought signatures) so they flow through to the assembled message

The compat bridge's three-tier lookup allows unregistered or community models to work with V3 streaming via best-effort parsing.
