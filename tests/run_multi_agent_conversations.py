"""Run multi-agent fixture conversations through the Boss-Worker system and report results.

Usage:
    uv run python tests/run_multi_agent_conversations.py
    uv run python tests/run_multi_agent_conversations.py --ids ma_conv_01 ma_conv_03
    uv run python tests/run_multi_agent_conversations.py --ids ma_conv_01 --console-trace
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

# Load .env and mirror SEC_LLM_OPENAI_API_KEY → OPENAI_API_KEY so the agents
# SDK tracer can export to platform.openai.com/traces.
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("SEC_LLM_OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["SEC_LLM_OPENAI_API_KEY"]

FIXTURES = Path(__file__).parent / "fixtures" / "test_multi_agent_conversations.json"

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

async def run_turn(
    message: str,
    history: list[dict],
) -> tuple[str, list[dict], list[str], dict]:
    """Run one turn via the Boss agent; return (answer, citations, tool_names, scratchpad)."""
    from agents import OpenAIProvider, RunConfig, Runner
    from sec_llm.agents.boss import boss_agent
    from sec_llm.dependencies import get_settings

    settings = get_settings()
    run_config = RunConfig(model_provider=OpenAIProvider(api_key=settings.openai_api_key))

    messages = history + [{"role": "user", "content": message}]
    result = await Runner.run(boss_agent, messages, run_config=run_config)

    answer = result.final_output or ""

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

    # Extract scratchpad from final output
    from sec_llm.runner import _extract_scratchpad
    scratchpad = _extract_scratchpad(result)

    return answer, citations, tool_names, scratchpad


def _check_turn(
    turn: dict,
    answer: str,
    tool_names: list[str],
    scratchpad: dict,
) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    failures: list[str] = []

    expected_tools = [tc["tool"] for tc in turn.get("expected_tool_calls", [])]

    # Check that every expected worker tool was called (order-insensitive)
    for exp in expected_tools:
        if exp not in tool_names:
            failures.append(f"Expected tool '{exp}' not called (got: {tool_names or ['none']})")

    # Check scratchpad when expected
    if turn.get("expect_scratchpad"):
        if not scratchpad:
            failures.append("Expected non-empty scratchpad in response, but none found")
        else:
            if "goal" not in scratchpad:
                failures.append("Scratchpad missing 'goal' field")
            if "tasks" not in scratchpad:
                failures.append("Scratchpad missing 'tasks' field")

    return failures


# ── conversation runner ───────────────────────────────────────────────────────

async def run_conversation(conv: dict) -> dict:
    """Run all turns of one multi-agent conversation fixture. Returns a result dict."""
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

    from agents import trace as agent_trace

    with agent_trace(
        workflow_name=f"[multi-agent test] {conv_id}: {description[:60]}",
        group_id=conv_id,
    ):
        for i, turn in enumerate(user_turns):
            message = turn["message"]
            expected_behavior = turn.get("expected_behavior", "")

            t0 = time.monotonic()
            try:
                answer, citations, tool_names, scratchpad = await run_turn(message, history)
                elapsed = time.monotonic() - t0
                failures = _check_turn(turn, answer, tool_names, scratchpad)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                failures = [f"Exception: {exc}"]
                answer = ""
                tool_names = []
                scratchpad = {}

            turn_result = {
                "turn": i + 1,
                "message": message,
                "answer": answer,
                "tool_names": tool_names,
                "scratchpad_goal": scratchpad.get("goal", "(none)"),
                "scratchpad_tasks": len(scratchpad.get("tasks", [])),
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
        color = _GREEN if turn_ok else _RED
        print(f"\n  {color}Turn {tr['turn']}{_RESET} ({tr['elapsed_s']}s)")
        print(f"    User:  {tr['message']}")
        print(f"    Tools: {tr['tool_names'] or ['(none)']}")
        print(f"    Scratchpad goal: {tr['scratchpad_goal']}")
        print(f"    Scratchpad tasks: {tr['scratchpad_tasks']}")
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

async def main(ids: list[str] | None, console_trace: bool = False) -> int:
    conversations = json.loads(FIXTURES.read_text())

    if console_trace:
        from agents import add_trace_processor
        from agents.tracing.processors import BatchTraceProcessor, ConsoleSpanExporter
        add_trace_processor(BatchTraceProcessor(ConsoleSpanExporter()))

    if ids:
        conversations = [c for c in conversations if c["id"] in ids]
        if not conversations:
            print(f"No conversations matched IDs: {ids}")
            return 1

    print(f"Running {len(conversations)} multi-agent conversation(s) …\n")
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
    parser = argparse.ArgumentParser(description="Run multi-agent fixture conversations against live Boss-Worker system")
    parser.add_argument("--ids", nargs="*", help="Specific conversation IDs to run (e.g. ma_conv_01 ma_conv_03)")
    parser.add_argument("--console-trace", action="store_true", help="Print span data to the terminal via ConsoleSpanExporter")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.ids, console_trace=args.console_trace)))
