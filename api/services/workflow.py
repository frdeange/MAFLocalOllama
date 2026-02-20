"""
Workflow Service
================
Runs the MAF SequentialBuilder pipeline and yields SSE events.
Translates MAF streaming events into the SSE event protocol.

Key design decisions:
- Buffers output per agent and emits ``agent_completed`` only once per agent
  (with the longest/best output), because MAF may fire multiple executor
  rounds for a single agent (e.g. tool-calling rounds in WeatherAnalyst).
- Strips ``=== PREVIOUS CONVERSATION CONTEXT ===`` markers that agents
  may echo back from the session context prefix.
- Caps at the 3 known agents; extra executor events are attributed to the
  last agent in the sequence.
"""

import json
import logging
import re
from typing import AsyncGenerator

from src.telemetry import trace_workflow
from src.workflows.travel_planner import build_travel_planner_workflow

logger = logging.getLogger(__name__)

# Agent names in execution order (must match SequentialBuilder participant order)
AGENT_SEQUENCE = ["Researcher", "WeatherAnalyst", "Planner"]

# Regex to strip echoed conversation-context markers from agent output
_CONTEXT_PREFIX_RE = re.compile(
    r"===\s*PREVIOUS CONVERSATION CONTEXT\s*===.*?===\s*END OF CONTEXT\s*===\s*",
    re.DOTALL,
)


def _sse_line(event: str, data: dict) -> str:
    """Format a single SSE event line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _extract_last_assistant_text(data) -> str:
    """Safely extract the last assistant message text from event data.

    MAF event.data can be:
        - A list of Message objects (each with .role and .text)
        - A list containing nested lists
        - A single Message object
        - A string
        - None
    """
    if data is None:
        return ""

    if isinstance(data, str):
        return data

    # If it's a list, flatten and iterate
    messages = data if isinstance(data, list) else [data]

    # Flatten nested lists
    flat = []
    for item in messages:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    # Try to extract assistant text from Message-like objects
    for msg in reversed(flat):
        try:
            role = getattr(msg, "role", None)
            text = getattr(msg, "text", None) or getattr(msg, "content", None)
            if role == "assistant" and text:
                return str(text)
        except Exception:
            continue

    # Last resort: stringify the last item
    if flat:
        last = flat[-1]
        if isinstance(last, str):
            return last
        text = getattr(last, "text", None) or getattr(last, "content", None)
        if text:
            return str(text)

    return ""


def _clean_output(text: str) -> str:
    """Strip context-prefix markers and filter trivial outputs.

    Returns an empty string for outputs that are meaningless (single char,
    bare punctuation, etc.) so callers can decide whether to show them.
    """
    if not text:
        return ""
    # Remove echoed context-prefix block
    text = _CONTEXT_PREFIX_RE.sub("", text)
    text = text.strip()
    # Filter trivially short / punctuation-only outputs
    if len(text) <= 1 or text in ("-", ".", "...", "—", "–"):
        return ""
    return text


async def run_workflow_sse(
    client: object,
    mcp_server_url: str,
    query: str,
) -> AsyncGenerator[str, None]:
    """Run the travel planner workflow and yield SSE-formatted events.

    This is the core bridge between MAF's SequentialBuilder streaming events
    and the SSE protocol consumed by the frontend.

    The function buffers MAF executor events and maps them to exactly 3 logical
    agent steps.  Each agent gets one ``agent_started`` and one
    ``agent_completed`` SSE event.  Extra executor rounds (tool-calling, etc.)
    are silently merged into the current agent's buffer.

    Args:
        client: OllamaChatClient instance.
        mcp_server_url: URL of the MCP server.
        query: The user's travel planning request (with optional context prefix).

    Yields:
        SSE-formatted strings (event + data lines).
    """
    try:
        workflow, mcp_tool = build_travel_planner_workflow(
            client=client,
            mcp_server_url=mcp_server_url,
        )

        yield _sse_line("workflow_started", {"workflow": "travel_planner"})

        with trace_workflow("travel_planner", query):
            async with mcp_tool:
                logger.info("MCP tool connected to %s", mcp_server_url)

                # ── state tracking ────────────────────────────────
                executor_count = 0                     # raw MAF executor events seen
                started_agents: set[str] = set()       # agents we emitted 'started' for
                completed_agents: set[str] = set()     # agents we emitted 'completed' for
                # Buffer: (agent_name, agent_idx, best_output_so_far)
                pending: tuple[str, int, str] | None = None
                last_meaningful_output = ""
                got_output_event = False

                async for event in workflow.run(query, stream=True):
                    logger.debug(
                        "MAF event: type=%s, data_type=%s",
                        event.type,
                        type(getattr(event, "data", None)),
                    )

                    # ── executor_invoked ──────────────────────────
                    if event.type == "executor_invoked":
                        executor_count += 1
                        idx = min(executor_count - 1, len(AGENT_SEQUENCE) - 1)
                        name = AGENT_SEQUENCE[idx]

                        # Transitioning to a different agent → flush pending
                        if pending and pending[0] != name:
                            p_name, p_idx, p_out = pending
                            if p_name not in completed_agents:
                                completed_agents.add(p_name)
                                if p_out:
                                    last_meaningful_output = p_out
                                yield _sse_line("agent_completed", {
                                    "agent": p_name,
                                    "step": p_idx + 1,
                                    "output": p_out,
                                })
                                logger.info(
                                    "Agent completed: %s (step %d, output_len=%d)",
                                    p_name, p_idx + 1, len(p_out),
                                )
                            pending = None

                        # Emit agent_started only once per agent
                        if name not in started_agents:
                            started_agents.add(name)
                            yield _sse_line("agent_started", {
                                "agent": name,
                                "step": idx + 1,
                            })
                            logger.info("Agent started: %s (step %d)", name, idx + 1)

                        # Initialise buffer for this agent (keep existing if same agent)
                        if pending is None or pending[0] != name:
                            pending = (name, idx, "")

                    # ── executor_completed ────────────────────────
                    elif event.type == "executor_completed":
                        idx = min(executor_count - 1, len(AGENT_SEQUENCE) - 1)
                        name = AGENT_SEQUENCE[idx]

                        raw = _extract_last_assistant_text(getattr(event, "data", None))
                        output = _clean_output(raw)

                        # Keep the longest (best) output for this agent
                        if pending and pending[0] == name:
                            if len(output) > len(pending[2]):
                                pending = (name, idx, output)

                        logger.debug(
                            "Executor completed: %s (raw_len=%d, clean_len=%d, best=%d)",
                            name, len(raw), len(output),
                            len(pending[2]) if pending else 0,
                        )

                    # ── output (workflow finished) ────────────────
                    elif event.type == "output":
                        got_output_event = True

                        # Flush last pending agent
                        if pending and pending[0] not in completed_agents:
                            p_name, p_idx, p_out = pending
                            completed_agents.add(p_name)
                            if p_out:
                                last_meaningful_output = p_out
                            yield _sse_line("agent_completed", {
                                "agent": p_name,
                                "step": p_idx + 1,
                                "output": p_out,
                            })
                            logger.info(
                                "Agent completed: %s (step %d, output_len=%d)",
                                p_name, p_idx + 1, len(p_out),
                            )

                        final = _clean_output(
                            _extract_last_assistant_text(getattr(event, "data", None))
                        )
                        if not final:
                            final = last_meaningful_output

                        yield _sse_line("workflow_completed", {
                            "final_output": final,
                        })
                        logger.info("Workflow completed (output_len=%d)", len(final))

                # Safety: if the loop ended without an 'output' event
                if not got_output_event:
                    logger.warning("No output event received — flushing remaining state")
                    if pending and pending[0] not in completed_agents:
                        p_name, p_idx, p_out = pending
                        completed_agents.add(p_name)
                        if p_out:
                            last_meaningful_output = p_out
                        yield _sse_line("agent_completed", {
                            "agent": p_name,
                            "step": p_idx + 1,
                            "output": p_out,
                        })
                    yield _sse_line("workflow_completed", {
                        "final_output": last_meaningful_output,
                    })

    except Exception as e:
        logger.exception("Workflow execution failed: %s", e)
        yield _sse_line("error", {"message": str(e)})
