"""
Workflow Service
================
Runs the MAF SequentialBuilder pipeline and yields SSE events.
Translates MAF streaming events into the SSE event protocol.
"""

import json
import logging
from typing import AsyncGenerator, cast

from agent_framework import Message as MAFMessage

from src.telemetry import trace_workflow
from src.workflows.travel_planner import build_travel_planner_workflow

logger = logging.getLogger(__name__)

# Agent names in execution order (must match SequentialBuilder participant order)
AGENT_SEQUENCE = ["Researcher", "WeatherAnalyst", "Planner"]


def _sse_line(event: str, data: dict) -> str:
    """Format a single SSE event line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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

                        # Extract the last assistant message as this agent's output
                        agent_output = ""
                        if hasattr(event, "data") and event.data:
                            messages = cast(list[MAFMessage], event.data)
                            for msg in reversed(messages):
                                if msg.role == "assistant" and msg.text:
                                    agent_output = msg.text
                                    break

                        last_agent_output = agent_output
                        yield _sse_line("agent_completed", {
                            "agent": agent_name,
                            "step": current_step,
                            "output": agent_output,
                        })
                        logger.info("Agent completed: %s (step %d)", agent_name, current_step)

                    elif event.type == "output":
                        # Final workflow output — extract the planner's response
                        final_output = ""
                        if hasattr(event, "data") and event.data:
                            messages = cast(list[MAFMessage], event.data)
                            for msg in reversed(messages):
                                if msg.role == "assistant" and msg.text:
                                    final_output = msg.text
                                    break

                        if not final_output:
                            final_output = last_agent_output

                        yield _sse_line("workflow_completed", {
                            "final_output": final_output,
                        })
                        logger.info("Workflow completed")

    except Exception as e:
        logger.exception("Workflow execution failed: %s", e)
        yield _sse_line("error", {"message": str(e)})
