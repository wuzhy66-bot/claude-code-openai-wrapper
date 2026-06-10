#!/usr/bin/env python3
"""Unit tests for client-side deferred tool mirroring."""

from src.models import ChatCompletionRequest, Message
from src import main


def _request(tools=None):
    return ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="use a tool")],
        defer_tools=True,
        tools=tools or [],
    )


def test_request_tool_specs_support_openai_function_tools(monkeypatch):
    monkeypatch.setattr(main, "_deferred_client_tool_profile_cache", [])

    req = _request(
        [
            {
                "type": "function",
                "function": {
                    "name": "PowerShell",
                    "description": "Run PowerShell",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
    )

    specs = main._extract_request_tool_specs(req)

    assert specs[0]["name"] == "PowerShell"
    assert specs[0]["input_schema"]["properties"]["command"]["type"] == "string"


def test_client_tool_mirror_maps_internal_mcp_names_to_client_names(monkeypatch):
    monkeypatch.setattr(main, "_deferred_tool_mirror_mode", "client")
    monkeypatch.setattr(main, "_deferred_client_tool_profile_cache", [])
    monkeypatch.setattr(main, "_deferred_client_platform", "")

    req = _request(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            }
        ]
    )

    mirror = main._build_client_tool_mirror(req)

    assert mirror is not None
    assert "mcp__client__Read" in mirror["allowed_tools"]
    assert mirror["internal_to_external"]["mcp__client__Read"] == "Read"
    assert main._map_deferred_tool_name("mcp__client__Read", mirror["internal_to_external"]) == "Read"


def test_windows_profile_adds_powershell_when_request_has_no_tools(monkeypatch):
    monkeypatch.setattr(main, "_deferred_tool_mirror_mode", "client")
    monkeypatch.setattr(main, "_deferred_client_tool_profile_cache", [])
    monkeypatch.setattr(main, "_deferred_client_platform", "windows")

    mirror = main._build_client_tool_mirror(_request())

    assert mirror is not None
    assert "PowerShell" in mirror["external_names"]
    assert mirror["internal_to_external"]["mcp__client__PowerShell"] == "PowerShell"
