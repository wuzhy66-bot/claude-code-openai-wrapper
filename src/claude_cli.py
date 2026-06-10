import os
import json
import tempfile
import atexit
import shutil
from typing import AsyncGenerator, Dict, Any, Optional, List
from pathlib import Path
import logging

from claude_agent_sdk import query, ClaudeAgentOptions

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _normalise_thinking_config(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    thinking_type = str(value.get("type") or "").strip().lower()
    if thinking_type == "adaptive":
        return {"type": "adaptive"}
    if thinking_type == "disabled":
        return {"type": "disabled"}
    if thinking_type == "enabled":
        try:
            budget_tokens = int(value.get("budget_tokens"))
        except (TypeError, ValueError):
            budget_tokens = 0
        if budget_tokens > 0:
            return {"type": "enabled", "budget_tokens": budget_tokens}
    return None


def _settings_with_thinking_summaries(settings: Optional[str]) -> str:
    settings_obj: Dict[str, Any] = {}
    if settings:
        settings_text = settings.strip()
        if settings_text.startswith("{") and settings_text.endswith("}"):
            try:
                settings_obj = json.loads(settings_text)
            except json.JSONDecodeError:
                logger.warning("Ignoring invalid JSON settings while enabling thinking summaries")
        else:
            settings_path = Path(settings_text)
            if settings_path.exists():
                try:
                    settings_obj = json.loads(settings_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    logger.warning("Ignoring unreadable settings file while enabling thinking summaries")
            else:
                logger.warning("Settings path not found while enabling thinking summaries: %s", settings)
    settings_obj["showThinkingSummaries"] = True
    return json.dumps(settings_obj)


def apply_claude_agent_reasoning_options(
    options: ClaudeAgentOptions,
    claude_options: Dict[str, Any],
) -> None:
    """Apply reasoning/thinking options to ClaudeAgentOptions in one place."""
    effort = claude_options.get("effort")
    if effort:
        options.effort = str(effort)

    thinking = _normalise_thinking_config(claude_options.get("thinking"))
    if thinking:
        options.thinking = thinking
    else:
        max_thinking_tokens = claude_options.get("max_thinking_tokens")
        if max_thinking_tokens is not None:
            try:
                options.max_thinking_tokens = int(max_thinking_tokens)
            except (TypeError, ValueError):
                logger.warning("Invalid max_thinking_tokens ignored: %s", max_thinking_tokens)

    settings = claude_options.get("settings")
    if isinstance(settings, str) and settings:
        options.settings = settings

    show_summaries = claude_options.get(
        "show_thinking_summaries",
        _env_flag("CLAUDE_WRAPPER_SHOW_THINKING_SUMMARIES", True),
    )
    if show_summaries:
        options.settings = _settings_with_thinking_summaries(options.settings)


class ClaudeCodeCLI:
    def __init__(self, timeout: int = 600000, cwd: Optional[str] = None):
        self.timeout = timeout / 1000  # Convert ms to seconds
        self.temp_dir = None

        # If cwd is provided (from CLAUDE_CWD env var), use it
        # Otherwise create an isolated temp directory
        if cwd:
            self.cwd = Path(cwd)
            # Check if the directory exists
            if not self.cwd.exists():
                logger.error(f"ERROR: Specified working directory does not exist: {self.cwd}")
                logger.error(
                    "Please create the directory first or unset CLAUDE_CWD to use a temporary directory"
                )
                raise ValueError(f"Working directory does not exist: {self.cwd}")
            else:
                logger.info(f"Using CLAUDE_CWD: {self.cwd}")
        else:
            # Create isolated temp directory (cross-platform)
            self.temp_dir = tempfile.mkdtemp(prefix="claude_code_workspace_")
            self.cwd = Path(self.temp_dir)
            logger.info(f"Using temporary isolated workspace: {self.cwd}")

            # Register cleanup function to remove temp dir on exit
            atexit.register(self._cleanup_temp_dir)

        # Import auth manager
        from src.auth import auth_manager, validate_claude_code_auth

        # Validate authentication
        is_valid, auth_info = validate_claude_code_auth()
        if not is_valid:
            logger.warning(f"Claude Code authentication issues detected: {auth_info['errors']}")
        else:
            logger.info(f"Claude Code authentication method: {auth_info.get('method', 'unknown')}")

        # Store auth environment variables for SDK
        self.claude_env_vars = auth_manager.get_claude_code_env_vars()

    async def verify_cli(self) -> bool:
        """Verify Claude Agent SDK is working and authenticated."""
        try:
            # Test SDK with a simple query
            logger.info("Testing Claude Agent SDK...")

            messages = []
            async for message in query(
                prompt="Hello",
                options=ClaudeAgentOptions(
                    max_turns=1,
                    cwd=self.cwd,
                    system_prompt={"type": "preset", "preset": "claude_code"},
                ),
            ):
                messages.append(message)
                # Break early on first response to speed up verification
                # Handle both dict and object types
                msg_type = (
                    getattr(message, "type", None)
                    if hasattr(message, "type")
                    else message.get("type") if isinstance(message, dict) else None
                )
                if msg_type == "assistant":
                    break

            if messages:
                logger.info("✅ Claude Agent SDK verified successfully")
                return True
            else:
                logger.warning("⚠️ Claude Agent SDK test returned no messages")
                return False

        except Exception as e:
            logger.error(f"Claude Agent SDK verification failed: {e}")
            logger.warning("Please ensure Claude Code is installed and authenticated:")
            logger.warning("  1. Install: npm install -g @anthropic-ai/claude-code")
            logger.warning("  2. Set ANTHROPIC_API_KEY environment variable")
            logger.warning("  3. Test: claude --print 'Hello'")
            return False

    async def run_completion(
        self,
        prompt: Any,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = True,
        max_turns: int = 10,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        permission_mode: Optional[str] = None,
        hooks: Optional[Dict[str, Any]] = None,
        resume: Optional[str] = None,
        effort: Optional[str] = None,
        max_thinking_tokens: Optional[int] = None,
        thinking: Optional[Dict[str, Any]] = None,
        settings: Optional[str] = None,
        show_thinking_summaries: Optional[bool] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run Claude Agent using the Python SDK and yield response chunks."""

        try:
            # Set authentication environment variables (if any)
            original_env = {}
            if self.claude_env_vars:  # Only set env vars if we have any
                for key, value in self.claude_env_vars.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value

            try:
                # Build SDK options
                options = ClaudeAgentOptions(max_turns=max_turns, cwd=self.cwd)

                # Set model if specified
                if model:
                    options.model = model
                apply_claude_agent_reasoning_options(
                    options,
                    {
                        "effort": effort,
                        "max_thinking_tokens": max_thinking_tokens,
                        "thinking": thinking,
                        "settings": settings,
                        "show_thinking_summaries": show_thinking_summaries,
                    },
                )

                # Set system prompt - CLAUDE AGENT SDK STRUCTURED FORMAT
                # Use structured format as per SDK documentation
                if system_prompt:
                    options.system_prompt = {"type": "text", "text": system_prompt}
                else:
                    # Use Claude Code preset to maintain expected behavior
                    options.system_prompt = {"type": "preset", "preset": "claude_code"}

                # Set tool restrictions
                if allowed_tools:
                    options.allowed_tools = allowed_tools
                if disallowed_tools:
                    options.disallowed_tools = disallowed_tools
                if hooks:
                    options.hooks = hooks

                # Set permission mode (needed for tool execution in API context)
                if permission_mode:
                    options.permission_mode = permission_mode

                # Handle session continuity
                if continue_session:
                    options.continue_session = True
                elif resume:
                    options.resume = resume
                elif session_id:
                    options.resume = session_id

                # Run the query and yield messages
                async for message in query(prompt=prompt, options=options):
                    # Debug logging
                    logger.debug(f"Raw SDK message type: {type(message)}")
                    logger.debug(f"Raw SDK message: {message}")

                    # Convert message object to dict if needed
                    if hasattr(message, "__dict__") and not isinstance(message, dict):
                        # Convert object to dict for consistent handling
                        message_dict = {}

                        # Get all attributes from the object
                        for attr_name in dir(message):
                            if not attr_name.startswith("_"):  # Skip private attributes
                                try:
                                    attr_value = getattr(message, attr_name)
                                    if not callable(attr_value):  # Skip methods
                                        message_dict[attr_name] = attr_value
                                except:
                                    pass

                        logger.debug(f"Converted message dict: {message_dict}")
                        yield message_dict
                    else:
                        yield message

            finally:
                # Restore original environment (if we changed anything)
                if original_env:
                    for key, original_value in original_env.items():
                        if original_value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = original_value

        except Exception as e:
            logger.error(f"Claude Agent SDK error: {e}")
            # Yield error message in the expected format
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": str(e),
            }

    def parse_claude_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the assistant message from Claude Agent SDK messages.

        Prioritizes ResultMessage.result for multi-turn conversations,
        falls back to last AssistantMessage content.
        """
        # First, check for ResultMessage with 'result' field (multi-turn completion)
        for message in messages:
            if message.get("subtype") == "success" and "result" in message:
                return message["result"]

        # Collect all text from AssistantMessages (take the last one with text)
        last_text = None
        for message in messages:
            # Look for AssistantMessage type (new SDK format)
            if "content" in message and isinstance(message["content"], list):
                text_parts = []
                for block in message["content"]:
                    # Handle TextBlock objects
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)

                if text_parts:
                    last_text = "\n".join(text_parts)

            # Fallback: look for old format
            elif message.get("type") == "assistant" and "message" in message:
                sdk_message = message["message"]
                if isinstance(sdk_message, dict) and "content" in sdk_message:
                    content = sdk_message["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Handle content blocks (Anthropic SDK format)
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        if text_parts:
                            last_text = "\n".join(text_parts)
                    elif isinstance(content, str):
                        last_text = content

        return last_text

    def extract_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata like costs, tokens, and session info from SDK messages."""
        metadata = {
            "session_id": None,
            "total_cost_usd": 0.0,
            "duration_ms": 0,
            "num_turns": 0,
            "model": None,
        }

        for message in messages:
            # New SDK format - ResultMessage
            if message.get("subtype") == "success" and "total_cost_usd" in message:
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
            # New SDK format - SystemMessage
            elif message.get("subtype") == "init" and "data" in message:
                data = message["data"]
                metadata.update({"session_id": data.get("session_id"), "model": data.get("model")})
            # Old format fallback
            elif message.get("type") == "result":
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
            elif message.get("type") == "system" and message.get("subtype") == "init":
                metadata.update(
                    {"session_id": message.get("session_id"), "model": message.get("model")}
                )

        return metadata

    def estimate_token_usage(
        self, prompt: str, completion: str, model: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Estimate token usage based on character count.

        Uses rough approximation: ~4 characters per token for English text.
        This is approximate and may not match actual tokenization.
        """
        # Rough approximation: 1 token ≈ 4 characters
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _cleanup_temp_dir(self):
        """Clean up temporary directory on exit."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary workspace: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {self.temp_dir}: {e}")
