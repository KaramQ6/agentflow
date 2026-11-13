"""Tool annotations under ``from __future__ import annotations`` (TOOL-1).

With the future import every annotation is a string. Pydantic can only resolve a name such
as ``Literal`` if it can see the function's globals, which a model built by ``create_model``
does not unless the annotations are resolved first.
"""

from __future__ import annotations

from typing import Literal, Optional

import pytest

from agentflow import tool

Direction = Literal["up", "down"]


@tool
async def move(direction: Literal["up", "down"], steps: int = 1) -> str:
    """Move in a direction."""
    return f"{direction} {steps}"


@tool
async def move_alias(direction: Direction, note: Optional[str] = None) -> str:  # noqa: UP007,UP045
    """Move, with a module-level alias and an Optional."""
    return f"{direction} {note}"


def test_a_literal_parameter_becomes_an_enum_in_the_schema():
    assert move.parameters["properties"]["direction"]["enum"] == ["up", "down"], (
        "a string annotation must resolve to the Literal, not fail or degrade to str (TOOL-1)"
    )


def test_a_module_level_alias_and_optional_resolve():
    assert move_alias.parameters["properties"]["direction"]["enum"] == ["up", "down"]


@pytest.mark.asyncio
async def test_the_literal_is_enforced_when_the_tool_is_called():
    assert await move.acall({"direction": "up", "steps": 2}) == "up 2"
    with pytest.raises(Exception, match="direction"):
        await move.acall({"direction": "sideways"})
