"""
security_audit.py
------------------
SentinelAI's "Security Mode" — a DEFENSIVE, LOCAL-ONLY self-audit of the host
this service is running on. It never scans, probes, or attacks any other
machine, and it never generates exploit/attack code. It only reads local
configuration and writes a human-readable report + a JSON report.

Checks performed (all read-only, all local):
  1. Flask debug mode (RCE risk if left on in production)
  2. CORS configuration (wildcard origins)
  3. Bind address (0.0.0.0 exposes the service to the whole network)
  4. Listening TCP ports on this host
  5. Outdated pip packages (known-vulnerable versions often lag behind)
  6. World-readable / world-writable sensitive files (.env, config files)
  7. Whether the process is running as root/admin (should not be, normally)
  8. Risky entries in ALLOWED_APPS (exec-kind entries, since those spawn
     local processes)

Usage:
    from modules.security_audit import run_audit
    report = run_audit(app_config={...}, allowed_apps={...})
"""

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(os.environ.get("SECURITY_REPORTS_DIR", "reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SENSITIVE_FILES = [".env", "config.py", "secrets.json"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_debug_mode(flask_debug: bool) -> dict:
    return {
        "check": "flask_debug_mode",
        "severity": "high" if flask_debug else "ok",
        "finding": "Flask debug=True is enabled (exposes the Werkzeug debugger, "
                   "which allows remote code execution)." if flask_debug
                   else "Flask debug mode is off.",
        "recommendation": "Set debug=False in production." if flask_debug else None,
    }


def _check_cors(cors_origins) -> dict:
    wildcard = cors_origins in ("*", None) or (
        isinstance(cors_origins, (list, tuple)) and "*" in cors_origins
    )
    return {
        "check": "cors_wildcard",
        "severity": "medium" if wildcard else "ok",
        "finding": "CORS allows all origins ('*'), so any website the operator's "
                   "browser visits can call this API." if wildcard
                   else "CORS is restricted to specific origins.",
        "recommendation": "Restrict CORS to the dashboard's actual origin." if wildcard else None,
    }


def _check_bind_address(host: str) -> dict:
    exposed = host in ("0.0.0.0", "::")
    return {
        "check": "bind_address",
        "severity": "medium" if exposed else "ok",
        "finding": f"Service binds to {host}, which is reachable from the whole "
                   f"local network, not just this machine." if exposed
                   else f"Service binds to {host} (localhost-only).",
        "recommendation": "Bind to 127.0.0.1 unless LAN access is intentional." if exposed else None,
    }


def _check_open_ports() -> dict:
    """Best-effort local port scan of THIS host only (loopback), read-only."""
    common_ports = [21, 22, 23, 25, 3389, 5000, 5001, 6379, 7860, 8000, 8080, 11434, 27017]
    open_ports = []
    for port in common_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                open_ports.append(port)
    return {
        "check": "open_local_ports",
        "severity": "info",
        "finding": f"Locally listening ports found: {open_ports}" if open_ports
                   else "No common ports found listening on localhost.",
        "recommendation": "Make sure every listed service is one you intentionally run.",
    }


def _check_outdated_packages() -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        outdated = json.loads(result.stdout or "[]")
    except Exception as exc:  # pip missing, timeout, etc.
        return {
            "check": "outdated_packages",
            "severity": "info",
            "finding": f"Could not check outdated packages: {exc}",
            "recommendation": None,
        }
    names = [f"{p['name']}=={p['version']}->{p['latest_version']}" for p in outdated]
    return {
        "check": "outdated_packages",
        "severity": "medium" if names else "ok",
        "finding": f"{len(names)} outdated package(s): {', '.join(names[:15])}" if names
                   else "All installed packages are up to date.",
        "recommendation": "Run `pip install -U <package>` for anything security-relevant." if names else None,
    }


def _check_file_permissions(base_dir: str) -> dict:
    findings = []
    for name in SENSITIVE_FILES:
        p = Path(base_dir) / name
        if not p.exists():
            continue
        mode = p.stat().st_mode
        world_readable = bool(mode & stat.S_IROTH)
        world_writable = bool(mode & stat.S_IWOTH)
        if world_readable or world_writable:
            findings.append(f"{p} is world-{'writable' if world_writable else 'readable'}")
    return {
        "check": "sensitive_file_permissions",
        "severity": "high" if findings else "ok",
        "finding": "; ".join(findings) if findings else "No world-readable/writable sensitive files found.",
        "recommendation": "chmod 600 the file(s) listed." if findings else None,
    }


def _check_running_as_root() -> dict:
    try:
        is_root = os.geteuid() == 0
    except AttributeError:
        is_root = False  # Windows
    return {
        "check": "running_as_root",
        "severity": "high" if is_root else "ok",
        "finding": "Process is running as root/UID 0." if is_root
                   else "Process is not running as root.",
        "recommendation": "Run as an unprivileged user." if is_root else None,
    }


def _check_allowed_apps(allowed_apps: dict) -> dict:
    exec_entries = [name for name, e in (allowed_apps or {}).items()
                     if isinstance(e, dict) and e.get("kind") == "exec"]
    return {
        "check": "allowed_apps_exec_entries",
        "severity": "low" if exec_entries else "ok",
        "finding": f"Exec-kind allow-list entries (spawn local processes): {exec_entries}" if exec_entries
                   else "No exec-kind allow-list entries configured.",
        "recommendation": "Keep this list minimal and reviewed — each entry is something "
                           "the voice/chat API can launch." if exec_entries else None,
    }


def run_audit(flask_debug: bool = False, cors_origins="*", bind_host: str = "0.0.0.0",
              base_dir: str = ".", allowed_apps: dict | None = None) -> dict:
    """Runs every check and writes a Markdown + JSON report to REPORTS_DIR.
    Returns the report dict (also what gets written as JSON)."""
    checks = [
        _check_debug_mode(flask_debug),
        _check_cors(cors_origins),
        _check_bind_address(bind_host),
        _check_open_ports(),
        _check_outdated_packages(),
        _check_file_permissions(base_dir),
        _check_running_as_root(),
        _check_allowed_apps(allowed_apps or {}),
    ]

    severity_rank = {"high": 3, "medium": 2, "low": 1, "info": 0, "ok": -1}
    checks.sort(key=lambda c: severity_rank.get(c["severity"], 0), reverse=True)

    report = {
        "generated_at": _now_iso(),
        "host": socket.gethostname(),
        "summary": {
            "high": sum(1 for c in checks if c["severity"] == "high"),
            "medium": sum(1 for c in checks if c["severity"] == "medium"),
            "low": sum(1 for c in checks if c["severity"] == "low"),
        },
        "checks": checks,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"security_audit_{ts}.json"
    md_path = REPORTS_DIR / f"security_audit_{ts}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"# SentinelAI Security Audit — {report['generated_at']}", "",
             f"Host: `{report['host']}`",
             f"High: {report['summary']['high']} · Medium: {report['summary']['medium']} · Low: {report['summary']['low']}",
             ""]
    for c in checks:
        lines.append(f"## [{c['severity'].upper()}] {c['check']}")
        lines.append(f"- Finding: {c['finding']}")
        if c["recommendation"]:
            lines.append(f"- Recommendation: {c['recommendation']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    report["report_files"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


if __name__ == "__main__":
    r = run_audit()
    print(json.dumps(r["summary"], indent=2))
    print("Report written to:", r["report_files"])
