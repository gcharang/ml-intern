#!/usr/bin/env python3
"""
PreToolUse hook — port of agent/core/agent_loop.py::_needs_approval.

Claude Code's static permission lists can't express ml-intern's
content-aware approval rules (e.g. "auto-approve CPU jobs but require
confirmation for GPU jobs"). This hook reads the tool input from stdin
and either:
  - exits 0 (allow without prompt) — equivalent to ml-intern auto-execute
  - prints a JSON `ask` decision so Claude Code prompts the user

Fail-safe: malformed payloads, non-dict tool_input, or empty tool_name
all result in `ask` (never silent allow). For an approval hook, falling
through to allow on error would defeat the policy.

Env knobs (hook-layer equivalents of fields in `agent.config.Config` —
the standalone CLI reads these from configs/main_agent_config.json):

  ML_INTERN_YOLO=1                 → skip ALL approvals    (Config.yolo_mode)
  ML_INTERN_CONFIRM_CPU_JOBS=0     → auto-approve CPU jobs (Config.confirm_cpu_jobs)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Mirror agent/tools/jobs_tool.py::CPU_FLAVORS
CPU_FLAVORS = ["cpu-basic", "cpu-upgrade"]

# Mirror agent/core/agent_loop.py::_IMMEDIATE_HF_JOB_RUNS (only "run"/"uv" get the
# CPU/GPU approval split — scheduled runs are gated separately and always prompt)
# and agent/tools/sandbox_tool.py::DEFAULT_CPU_SANDBOX_HARDWARE (the default CPU
# sandbox is auto-approved; only non-default hardware needs confirmation).
IMMEDIATE_HF_JOB_RUNS = ("run", "uv")
DEFAULT_CPU_SANDBOX_HARDWARE = "cpu-basic"


def _normalize_operation(operation: Any) -> str:
    """Mirror agent/core/approval_policy.py::normalize_tool_operation."""
    return str(operation or "").strip().lower()


def _is_scheduled_operation(operation: Any) -> bool:
    """Mirror agent/core/approval_policy.py::is_scheduled_operation."""
    return _normalize_operation(operation).startswith("scheduled ")


def _short_name(tool_name: str) -> str:
    """MCP tools surface in Claude Code as `mcp__<server>__<tool>`; strip the prefix."""
    return tool_name.split("__")[-1] if tool_name.startswith("mcp__") else tool_name


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _check_training_script_save_pattern(script: str) -> str | None:
    """Inspired by agent/utils/reliability_checks.py::check_training_script_save_pattern.

    Returns a warning when an hf_jobs script appears to load a model but
    not push it back to the Hub (job storage is ephemeral — the model is
    lost when the job ends). Source also emits a green "will be pushed"
    confirmation; we drop that — hook output is shown only when forcing
    a prompt, and a positive note there would be noise.
    """
    if not isinstance(script, str):
        return None
    has_from_pretrained = "from_pretrained" in script
    has_push_to_hub = "push_to_hub" in script
    if has_from_pretrained and not has_push_to_hub:
        return "WARNING: training script loads a model with `from_pretrained` but has no `push_to_hub` call — the trained model will be lost when the job ends."
    return None


def _hf_jobs_script_warning(tool_input: dict) -> str | None:
    """Extract the script body from an hf_jobs invocation and run save-pattern check."""
    operation = _normalize_operation(tool_input.get("operation"))
    if operation not in IMMEDIATE_HF_JOB_RUNS and not _is_scheduled_operation(
        operation
    ):
        return None
    script = (
        tool_input.get("script")
        or tool_input.get("uv_script")
        or tool_input.get("source")
        or ""
    )
    return _check_training_script_save_pattern(script)


def _needs_approval(tool_name: str, tool_input: dict) -> bool:
    """Port of agent/core/agent_loop.py::_needs_approval (+ _base_needs_approval).

    Ordering mirrors the source and is load-bearing: scheduled HF jobs are
    checked BEFORE the YOLO bypass, because scheduled (recurring/unbounded) jobs
    always require manual confirmation — even under YOLO. Everything else is
    bypassed by YOLO, then evaluated by the per-tool rules.

    Diverges from source in one place: source short-circuits to False on
    malformed args via `_validate_tool_args` so a downstream validation error
    surfaces. Here we don't have that path — Claude Code validates input
    shape against the MCP schema upstream, so any payload reaching this hook
    is already structurally valid.

    Mirrors the sync predicate `_needs_approval`; the agent's runtime path is
    the async `_approval_decision`, whose only extra logic — session-scoped
    cost-cap budgeting — has no stateless-hook equivalent and is out of scope.
    """
    short_name = _short_name(tool_name)

    # Scheduled HF jobs ALWAYS require approval — even under YOLO. Mirrors
    # agent_loop._needs_approval, where _is_scheduled_hf_job_run precedes the
    # config.yolo_mode bypass. (Auto-approving a recurring job is the unsafe
    # direction for an approval hook.)
    if short_name == "hf_jobs" and _is_scheduled_operation(tool_input.get("operation")):
        return True

    if _env_flag("ML_INTERN_YOLO", False):
        return False

    if short_name == "sandbox_create":
        # Only non-default hardware needs approval; the default CPU sandbox is
        # auto-approved. Mirrors agent_loop._base_needs_approval +
        # sandbox_tool.DEFAULT_CPU_SANDBOX_HARDWARE.
        hardware = tool_input.get("hardware") or DEFAULT_CPU_SANDBOX_HARDWARE
        return hardware != DEFAULT_CPU_SANDBOX_HARDWARE

    if short_name == "hf_jobs":
        # Scheduled runs handled above; only immediate "run"/"uv" reach the
        # CPU/GPU split. Any other operation (logs, cancel, status, ...) is
        # non-destructive and auto-approved.
        operation = _normalize_operation(tool_input.get("operation"))
        if operation not in IMMEDIATE_HF_JOB_RUNS:
            return False

        hardware_flavor = (
            tool_input.get("hardware_flavor")
            or tool_input.get("flavor")
            or tool_input.get("hardware")
            or "cpu-basic"
        )
        is_cpu_job = hardware_flavor in CPU_FLAVORS

        if is_cpu_job:
            return _env_flag("ML_INTERN_CONFIRM_CPU_JOBS", True)

        return True  # GPU jobs always prompt

    # Note: hf_private_repos is intentionally not handled. agent/core/tools.py
    # disables it ("replaced by hf_repo_files and hf_repo_git"). The two
    # rules below cover the same destructive operations on the live tools.

    if short_name == "hf_repo_files":
        operation = _normalize_operation(tool_input.get("operation"))
        if operation in ("upload", "delete"):
            return True

    if short_name == "hf_repo_git":
        operation = _normalize_operation(tool_input.get("operation"))
        if operation in (
            "delete_branch",
            "delete_tag",
            "merge_pr",
            "create_repo",
            "update_repo",
        ):
            return True

    return False


def _ask(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        # Fail-safe: a malformed payload to an APPROVAL hook must not silently
        # allow the tool. Log to stderr so the failure is inspectable.
        print(
            f"[ml-intern] approval hook: malformed stdin ({e}); forcing prompt",
            file=sys.stderr,
        )
        print(
            json.dumps(
                _ask(
                    "ml-intern: approval hook received malformed input — confirm before proceeding"
                )
            )
        )
        return 0

    if not isinstance(payload, dict):
        print(
            f"[ml-intern] approval hook: stdin is {type(payload).__name__}, expected dict; forcing prompt",
            file=sys.stderr,
        )
        print(
            json.dumps(
                _ask(
                    "ml-intern: approval hook received unexpected input — confirm before proceeding"
                )
            )
        )
        return 0

    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        print(
            f"[ml-intern] approval hook: tool_input is {type(tool_input).__name__}, expected dict; forcing prompt",
            file=sys.stderr,
        )
        print(
            json.dumps(
                _ask(
                    f"ml-intern: {tool_name or 'tool'} received non-dict input — confirm before proceeding"
                )
            )
        )
        return 0

    if not tool_name:
        print(
            "[ml-intern] approval hook: empty tool_name; forcing prompt",
            file=sys.stderr,
        )
        print(
            json.dumps(
                _ask(
                    "ml-intern: approval hook received empty tool_name — confirm before proceeding"
                )
            )
        )
        return 0

    needs = _needs_approval(tool_name, tool_input)

    # Reliability warnings ride along — surface them by forcing a prompt
    # even when the rule would otherwise auto-approve.
    short_name = _short_name(tool_name)
    warning: str | None = None
    if short_name == "hf_jobs":
        warning = _hf_jobs_script_warning(tool_input)
        if warning:
            needs = True

    if needs:
        reason_bits = [
            f"ml-intern policy: {tool_name} requires user confirmation "
            f"(see .claude/hooks/pre_tool_use_approval.py)"
        ]
        if warning:
            reason_bits.append(warning)
        print(json.dumps(_ask(" | ".join(reason_bits))))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
