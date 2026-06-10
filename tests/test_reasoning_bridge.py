from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk.types import ThinkingBlock

from src import main
from src.claude_cli import apply_claude_agent_reasoning_options
from src.models import ChatCompletionRequest, Message


def test_reasoning_summary_from_bridge_chunks_extracts_thinking_blocks():
    chunks = [{"content": [ThinkingBlock(thinking="Checked constraints.", signature="sig")]}]

    assert main._reasoning_summary_from_bridge_chunks(chunks) == "Checked constraints."


def test_chat_request_maps_effort_to_adaptive_thinking():
    request = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="hello")],
        reasoning_effort="high",
    )

    options = request.to_claude_options()

    assert options["effort"] == "high"
    assert options["thinking"] == {"type": "adaptive"}


def test_chat_request_preserves_explicit_thinking_config():
    request = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="hello")],
        reasoning_effort="high",
        thinking={"type": "enabled", "budget_tokens": 4096},
    )

    assert request.to_claude_options()["thinking"] == {
        "type": "enabled",
        "budget_tokens": 4096,
    }


def test_apply_reasoning_options_sets_sdk_fields_and_summary_setting():
    options = ClaudeAgentOptions()

    apply_claude_agent_reasoning_options(
        options,
        {
            "effort": "max",
            "thinking": {"type": "adaptive"},
            "show_thinking_summaries": True,
        },
    )

    assert options.effort == "max"
    assert options.thinking == {"type": "adaptive"}
    assert '"showThinkingSummaries": true' in options.settings
