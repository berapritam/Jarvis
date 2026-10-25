"""
skills_engine.py
-----------------
"Automated skill generation" + "self-thinking" for SentinelAI.

Design choice (important): generated skills are written to disk as plain
.py files under SKILLS_DIR for the OPERATOR to review and import — this
module never exec()/eval()s model output automatically. Letting an LLM's
output run unreviewed on your machine is how a local assistant turns into
a remote-code-execution hole, so skill drafts always land as files first.

Two pieces:
  1. generate_skill(goal)  -> asks Ollama to draft a Python function that
     accomplishes `goal`, saves it as skills/<slug>.py with a metadata
     header, returns the file path + code.
  2. reflect(conversation_log, notifications) -> asks Ollama to look at
     recent activity and propose 1-3 next-step suggestions (plain text,
     not code) — this is the "self-thinking" loop. It's a suggestion
     queue, not an autonomous actor: nothing fires without a human
     approving it via /api/skills/approve.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("sentinelai.skills")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", "skills"))
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

SKILL_SYSTEM_PROMPT = (
    "You write a single, self-contained Python function that accomplishes the "
    "user's stated goal. Only standard library or requests/flask. No shell "
    "commands, no file deletion, no network calls except to localhost. "
    "Output ONLY the Python code, no explanation, no markdown fences."
)

REFLECT_SYSTEM_PROMPT = (
    "You are reviewing recent assistant activity for a personal dashboard "
    "called SentinelAI. Suggest at most 3 short, concrete next steps the "
    "operator might want (plain language, one line each, no code). "
    "If nothing stands out, say so."
)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:50] or "skill"


def _ollama_generate(system_prompt: str, user_prompt: str, timeout: int = 60) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def generate_skill(goal: str) -> dict:
    """Drafts a skill for `goal` and saves it to SKILLS_DIR for review.
    Does NOT import or execute the generated code."""
    if not goal or not goal.strip():
        return {"error": "'goal' is required"}

    code = _ollama_generate(SKILL_SYSTEM_PROMPT, goal)
    # Strip accidental markdown fences if the model adds them anyway.
    code = re.sub(r"^```(?:python)?\n?|```$", "", code.strip(), flags=re.MULTILINE).strip()

    slug = _slugify(goal)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{slug}.py"
    path = SKILLS_DIR / filename

    header = (
        f'"""\n'
        f"AUTO-GENERATED DRAFT — NOT executed automatically.\n"
        f"Goal: {goal}\n"
        f"Generated: {ts}\n"
        f"Review before importing or running this file.\n"
        f'"""\n\n'
    )
    path.write_text(header + code + "\n", encoding="utf-8")

    log.info("Drafted skill '%s' -> %s", slug, path)
    return {
        "goal": goal,
        "file": str(path),
        "status": "draft_saved_pending_review",
        "code_preview": code[:500],
    }


def list_skills() -> list[dict]:
    out = []
    for p in sorted(SKILLS_DIR.glob("*.py")):
        out.append({"file": str(p), "modified": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()})
    return out


def reflect(conversation_log: list[dict], notifications: list[dict]) -> dict:
    """'Self-thinking': summarizes recent activity into a few suggestions.
    Returns text suggestions only — never auto-acts on them."""
    recent_chat = conversation_log[-10:]
    recent_notes = notifications[-10:]
    prompt = (
        f"Recent conversation (last {len(recent_chat)} turns): "
        f"{json.dumps(recent_chat, ensure_ascii=False)}\n\n"
        f"Recent notifications (last {len(recent_notes)}): "
        f"{json.dumps(recent_notes, ensure_ascii=False)}"
    )
    try:
        text = _ollama_generate(REFLECT_SYSTEM_PROMPT, prompt)
    except requests.RequestException as exc:
        return {"error": f"Ollama unreachable: {exc}"}

    suggestions = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "suggestions": suggestions[:3]}
