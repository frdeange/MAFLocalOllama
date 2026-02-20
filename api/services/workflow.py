"""
Workflow Service
================
Runs the MAF SequentialBuilder pipeline and yields SSE events.
Translates MAF streaming events into the SSE event protocol.
"""

import json
import logging
from typing import AsyncGenerator

from src.telemetry import trace_workflow
from src.workflows.travel_planner import build_travel_planner_workflow

logger = logging.getLogger(__name__)

# Agent names in execution order (must match SequentialBuilder participant order)
AGENT_SEQUENCE = ["Researcher", "WeatherAnalyst", "Planner"]


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


async def run_workflow_sse(
    client: object,
    mcp_server_url: str,
    query: str,
) -> AsyncGenerator[str, None]:
    """Run the travel planner workflow and yield SSE-formatted events.

    This is the core bridge between MAF's SequentialBuilder streaming events
    and the SSE protocol consumed by the frontend.

    Args:
        client: OllamaChatClient instance.
        mcp_server_url: URL of the MCP server.
        query: The user's travel planning request (with optional context prefix).

    Yields:
        SSE-formatted strings (event + data lines).
    """
    try:
        # Build the workflow
        workflow, mcp_tool = build_travel_planner_workflow(
            client=client,
            mcp_server_url=mcp_server_url,
        )

        # Emit workflow started
        yield _sse_line("workflow_started", {"workflow": "travel_planner"})

        with trace_workflow("travel_planner", query):
            async with mcp_tool:
                logger.info("MCP tool connected to %s", mcp_server_url)

                # Track agent progression
                current_step = 0
                last_agent_output = ""

                async for event in workflow.run(query, stream=True):
                    logger.debug("MAF event: type=%s, data_type=%s", event.type, type(getattr(event, 'data', None)))

                    if event.type == "executor_invoked":
                        current_step += 1
                        agent_name = AGENT_SEQUENCE[min(current_step - 1, len(AGENT_SEQUENCE) - 1)]
                        yield _sse_line("agent_started", {
                            "agent": agent_name,
                            "step": current_step,
                        })
                        logger.info("Agent started: %s (step %d)", agent_name, current_step)

                    elif event.type == "executor_completed":
                        agent_name = AGENT_SEQUENCE[min(current_step - 1, len(AGENT_SEQUENCE) - 1)]

                        agent_output = ""
                        if hasattr(event, "data") and event.data:
                            agent_output = _extract_last_assistant_text(event.data)

                        last_agent_output = agent_output
                        yield _sse_line("agent_completed", {
                            "agent": agent_name,
                            "step": current_step,
                            "output": agent_output,
                        })
                        logger.info("Agent completed: %s (step %d, output_len=%d)",
                                    agent_name, current_step, len(agent_output))

                    elif event.type == "output":
                        final_output = ""
                        if hasattr(event, "data") and event.data:
                            final_output = _extract_last_assistant_text(event.data)

                        if not final_output:
                            final_output = last_agent_output

                        yield _sse_line("workflow_completed", {
                            "final_output": final_output,
                        })
                        logger.info("Workflow completed (output_len=%d)", len(final_output))

    except Exception as e:
        logger.exception("Workflow execution failed: %s", e)
        yield _sse_line("error", {"message": str(e)})
