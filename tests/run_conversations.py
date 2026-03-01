"""Run fixture conversations through the live agent and report results.

Usage:
    uv run python tests/run_conversations.py
    uv run python tests/run_conversations.py --ids conv_01 conv_04 conv_07
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path

# Ensure the src package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

FIXTURES = Path(__file__).parent / "fixtures" / "test_conversations.json"

# ── helpers ──────────────────────────────────────────────────────────────────

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _ok(msg: str) -> str:  return f"{_GREEN}✓{_RESET} {msg}"
def _fail(msg: str) -> str: return f"{_RED}✗{_RESET} {msg}"
def _warn(msg: str) -> str: return f"{_YELLOW}!{_RESET} {msg}"
def _hdr(msg: str) -> str:  return f"\n{_BOLD}{msg}{_RESET}"


# ── core runner ──────────────────────────────────────────────────────────────

async def run_turn(message: str, history: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Run one turn; return (answer, citations, tool_names_called)."""
    from agents import OpenAIProvider, RunConfig, Runner
    from sec_llm.agent import sec_agent
    from sec_llm.dependencies import get_settings

    settings = get_settings()
    run_config = RunConfig(model_provider=OpenAIProvider(api_key=settings.openai_api_key))

    messages = history + [{"role": "user", "content": message}]
    result = await Runner.run(sec_agent, messages, run_config=run_config)

    answer = result.final_output or ""

    # Collect tool names that were called this turn.
    # ToolCallItem: .raw_item is a ResponseFunctionToolCall with .name
    # ToolCallOutputItem: .output is a string representation of the result dict
    tool_names: list[str] = []
    citations: list[dict] = []
    for item in result.new_items:
        item_type = getattr(item, "type", None)
        if item_type == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            name = getattr(raw, "name", None) or getattr(item, "name", "unknown")
            tool_names.append(name)
        elif item_type == "tool_call_output_item":
            raw = getattr(item, "output", None)
            if raw:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(data, dict):
                        meta = data.get("metadata")
                        if meta:
                            citations.append(meta)
                except Exception:
                    pass

    return answer, citations, tool_names


def _check_turn(turn: dict, answer: str, tool_names: list[str]) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    failures: list[str] = []

    expected_tools = [tc["tool"] for tc in turn.get("expected_tool_calls", [])]

    # Check that every expected tool was called (order-insensitive)
    for exp in expected_tools:
        if exp not in tool_names:
            failures.append(f"Expected tool '{exp}' not called (got: {tool_names or ['none']})")

    # Check that no unexpected tools were called when none expected
    if not expected_tools and tool_names:
        failures.append(f"Expected NO tool calls, but agent called: {tool_names}")

    return failures


# ── conversation runner ───────────────────────────────────────────────────────

async def run_conversation(conv: dict) -> dict:
    """Run all turns of one conversation fixture. Returns a result dict."""
    conv_id = conv["id"]
    description = conv["description"]
    turns = conv["turns"]

    result = {
        "id": conv_id,
        "description": description,
        "turn_results": [],
        "passed": True,
    }

    history: list[dict] = []
    user_turns = [t for t in turns if t["role"] == "user"]

    for i, turn in enumerate(user_turns):
        message = turn["message"]
        expected_behavior = turn.get("expected_behavior", "")

        t0 = time.monotonic()
        try:
            answer, citations, tool_names = await run_turn(message, history)
            elapsed = time.monotonic() - t0
            failures = _check_turn(turn, answer, tool_names)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            exc_str = str(exc)
            # A triggered guardrail tripwire is the expected outcome for out-of-scope
            # queries (conv_06). Treat it as a pass when no tool calls are expected.
            expected_tools = [tc["tool"] for tc in turn.get("expected_tool_calls", [])]
            if "tripwire" in exc_str.lower() and not expected_tools:
                answer = f"[Guardrail blocked: {exc_str}]"
                tool_names = []
                failures = []
            else:
                failures = [f"Exception: {exc}"]
                answer = ""
                tool_names = []

        turn_result = {
            "turn": i + 1,
            "message": message,
            "answer": answer,
            "tool_names": tool_names,
            "expected_behavior": expected_behavior,
            "failures": failures,
            "elapsed_s": round(elapsed, 1),
        }
        result["turn_results"].append(turn_result)

        if failures:
            result["passed"] = False

        # Append this turn + assistant response to history for multi-turn conversations
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

    return result


# ── reporting ─────────────────────────────────────────────────────────────────

def print_result(res: dict) -> None:
    status = _ok("PASS") if res["passed"] else _fail("FAIL")
    print(_hdr(f"[{res['id']}] {res['description']}"))
    print(f"  Status: {status}")

    for tr in res["turn_results"]:
        turn_ok = not tr["failures"]
        icon = "✓" if turn_ok else "✗"
        color = _GREEN if turn_ok else _RED
        print(f"\n  {color}Turn {tr['turn']}{_RESET} ({tr['elapsed_s']}s)")
        print(f"    User:  {tr['message']}")
        print(f"    Tools: {tr['tool_names'] or ['(none)']}")
        # Wrap answer for readability
        wrapped = textwrap.fill(tr["answer"], width=90, initial_indent="    Answer: ",
                                subsequent_indent="           ")
        print(wrapped)
        print(f"    {_YELLOW}Expected:{_RESET} {tr['expected_behavior']}")

        for f in tr["failures"]:
            print(f"    {_fail(f)}")


def print_summary(results: list[dict]) -> None:
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    color = _GREEN if passed == total else _RED
    print(_hdr(f"═══ Summary: {color}{passed}/{total} conversations passed{_RESET} ═══"))
    for r in results:
        icon = _ok(r["id"]) if r["passed"] else _fail(r["id"])
        print(f"  {icon}")


# ── entry point ───────────────────────────────────────────────────────────────

async def main(ids: list[str] | None) -> int:
    # Load .env manually so the script works without uvicorn
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    conversations = json.loads(FIXTURES.read_text())

    if ids:
        conversations = [c for c in conversations if c["id"] in ids]
        if not conversations:
            print(f"No conversations matched IDs: {ids}")
            return 1

    print(f"Running {len(conversations)} conversation(s) …\n")
    all_results: list[dict] = []

    for conv in conversations:
        print(f"  → {conv['id']}: {conv['description'][:70]}…")
        res = await run_conversation(conv)
        all_results.append(res)

    print("\n" + "═" * 60)
    for res in all_results:
        print_result(res)
    print_summary(all_results)

    return 0 if all(r["passed"] for r in all_results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fixture conversations against live agent")
    parser.add_argument("--ids", nargs="*", help="Specific conversation IDs to run (e.g. conv_01 conv_04)")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.ids)))
