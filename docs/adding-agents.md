# Adding New Agents

## Overview

This guide explains how to add new agents to the Travel Planner orchestration.
All agents follow the **factory function** pattern and are registered in the
`src/agents/` package.

## Step-by-Step

### 1. Create the Agent Module

Create a new file in `src/agents/`, e.g. `src/agents/budget_analyst.py`:

```python
"""
Budget Analyst Agent
====================
Estimates travel costs and creates budget breakdowns.
"""

from agent_framework import Agent


def create_budget_analyst_agent(client: object) -> Agent:
    """Create the Budget Analyst agent.

    Args:
        client: A FoundryLocalClient (or any client implementing as_agent).

    Returns:
        A configured Agent instance.
    """
    return client.as_agent(
        name="BudgetAnalyst",
        instructions=(
            "You are a travel budget analyst. Based on the destination, "
            "provide a detailed budget breakdown including:\n"
            "1. Accommodation estimates (budget/mid/luxury)\n"
            "2. Daily food costs\n"
            "3. Transportation within the city\n"
            "4. Attraction entry fees\n"
            "Output in USD with clear categories."
        ),
    )
```

### 2. Register in the Package

Add the factory to `src/agents/__init__.py`:

```python
from .budget_analyst import create_budget_analyst_agent

__all__ = [
    # ... existing exports
    "create_budget_analyst_agent",
]
```

### 3. Wire into a Workflow

Add the agent as a participant in `src/workflows/travel_planner.py`:

```python
from ..agents import create_budget_analyst_agent

budget_analyst = create_budget_analyst_agent(client)

workflow = SequentialBuilder(
    participants=[researcher, weather_analyst, budget_analyst, planner],
).build()
```

**Important**: The order in the `participants` list determines the execution order.
Each agent receives the full conversation from all previous agents.

### 4. Agent with MCP Tools

If your agent needs external tools, follow the Weather Analyst pattern:

```python
from agent_framework import Agent, MCPStreamableHTTPTool


def create_my_agent(client: object, mcp_server_url: str) -> tuple[Agent, MCPStreamableHTTPTool]:
    mcp_tool = MCPStreamableHTTPTool(
        name="my_tools",
        url=mcp_server_url,
        description="Description of tool capabilities",
    )

    agent = client.as_agent(
        name="MyAgent",
        instructions="...",
        tools=[mcp_tool],
    )

    return agent, mcp_tool
```

**Remember**: The `MCPStreamableHTTPTool` must be used as an async context manager
(`async with mcp_tool:`) before the workflow runs.

### 5. Add Tests

Create `tests/test_my_agent.py` to validate:
- The factory function exists and is callable
- The agent uses `as_agent()` pattern
- Tool-based agents use `MCPStreamableHTTPTool`

See `tests/test_architecture.py` for reference patterns.

## Conventions

| Convention | Rule |
|-----------|------|
| Module name | Lowercase, descriptive: `budget_analyst.py` |
| Factory function | `create_<name>_agent(client, ...)` |
| Agent name | PascalCase in `as_agent(name="BudgetAnalyst")` |
| Instructions | Clear, structured prompt with numbered items |
| Tools | Only via MCP (no local function tools in agents) |
| Return type | `Agent` (no tools) or `tuple[Agent, MCPStreamableHTTPTool]` (with tools) |
