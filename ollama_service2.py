"""
SentinelAI - Ollama AI Assistant Microservice
------------------------------------------------
A small standalone Flask service that sits in front of a locally running
Ollama instance and exposes a simple HTTP API for the SentinelAI dashboard's
AI assistant panel.

Model: llama3.2:3b   (pull it first: `ollama pull llama3.2:3b`)

Run:
    pip install -r requirements.txt
    ollama serve                      # in one terminal (Ollama itself)
    python ollama_service.py          # in another terminal (this bridge)

Server mode:
    The assistant can be toggled ON/OFF via GET/POST /api/server-mode,
    mirroring the "Auto-scan toggle" pattern already used in the FastAPI
    backend (see main.py -> run_auto_scan / SystemSettings.auto_scan_enabled).
    When server mode is OFF, /api/chat and /api/chat/stream reject requests
    with 503 so the frontend can show "AI assistant offline" instead of
    silently hammering Ollama.
"""

import os
import json
import base64
import logging
import subprocess
import webbrowser
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template, request, Response, stream_with_context
from flask_cors import CORS

from modules.face_detection import detect_faces_b64
from modules.skills_engine import generate_skill, list_skills, reflect
from modules.security_audit import run_audit
from modules.elevenlabs_tts import synthesize as tts_synthesize, list_voices as tts_list_voices, TTSError

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
REQUEST_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
PORT = int(os.environ.get("OLLAMA_SERVICE_PORT", "5001"))

# Image generation backend (e.g. Automatic1111 / SD WebUI style API).
IMAGE_GEN_HOST = os.environ.get("IMAGE_GEN_HOST", "http://localhost:7860")
IMAGE_GEN_TIMEOUT = int(os.environ.get("IMAGE_GEN_TIMEOUT", "120"))
IMAGE_OUTPUT_DIR = os.environ.get("IMAGE_OUTPUT_DIR", "generated_images")

# --------------------------------------------------------------------------
# App launcher allow-list
# --------------------------------------------------------------------------
# SECURITY: this intentionally does NOT accept arbitrary commands/paths from
# the request body, and it does NOT run as root. This service has CORS wide
# open (origins="*") and binds to 0.0.0.0, so any endpoint that ran
# caller-supplied commands (or ran with elevated privileges) would be a
# remote-code-execution hole reachable from any website the operator's
# browser visits. Instead, the operator defines a fixed name -> action
# allow-list here (or via ALLOWED_APPS_JSON), and the API can only ever
# trigger one of those exact, pre-approved actions, as whatever unprivileged
# user started this process.
#
# Each entry is one of:
#   {"kind": "url", "target": "https://..."}                — opens the
#       person's default browser to a fixed URL. This is the safest way to
#       give "open WhatsApp / Facebook / Instagram" behavior, since it works
#       across OSes without needing a specific desktop client installed and
#       never touches subprocess/exec.
#   {"kind": "exec", "cmd": ["executable", "arg1", ...]}     — launches a
#       specific local executable (e.g. a desktop app installed on this
#       machine). Only used for apps you've explicitly configured.
#
# Configure via env var, e.g.:
#   ALLOWED_APPS_JSON='{
#     "whatsapp":  {"kind": "url",  "target": "https://web.whatsapp.com"},
#     "notepad":   {"kind": "exec", "cmd": ["notepad.exe"]}
#   }'
_default_allowed_apps = {
    # Pre-loaded, commonly requested apps. These open the web version in the
    # operator's default browser rather than assuming a specific desktop
    # client path — swap in {"kind":"exec","cmd":[...]} for any of these if
    # you have the actual desktop app installed and want that instead.
    "whatsapp":   {"kind": "url", "target": "https://web.whatsapp.com"},
    "facebook":   {"kind": "url", "target": "https://www.facebook.com"},
    "instagram":  {"kind": "url", "target": "https://www.instagram.com"},
}
try:
    _configured = json.loads(os.environ.get("ALLOWED_APPS_JSON", ""))
except json.JSONDecodeError:
    _configured = None
    log_startup_warning = bool(os.environ.get("ALLOWED_APPS_JSON"))
else:
    log_startup_warning = False

ALLOWED_APPS = _configured if _configured else _default_allowed_apps

# Backward-compat: allow the older flat-list form too, e.g. ["notepad.exe"],
# by normalizing it into the {"kind": "exec", ...} shape used below.
for _name, _entry in list(ALLOWED_APPS.items()):
    if isinstance(_entry, list):
        ALLOWED_APPS[_name] = {"kind": "exec", "cmd": _entry}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentinelai.ollama")

if log_startup_warning:
    log.warning("ALLOWED_APPS_JSON was set but not valid JSON — falling back to the empty default allow-list.")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/health": {"origins": "*"}})  # tighten in production
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# In-memory "server mode" state
# --------------------------------------------------------------------------
# NOTE: in-memory only -> resets when the process restarts. If you need this
# to persist, back it with the same DB the FastAPI backend uses
# (models.SystemSettings), the same way auto_scan_enabled works in main.py.

_state = {
    "server_mode": False,      # AI assistant is OFF by default
    "last_toggled_at": None,
    "conversation_log": [],    # simple in-memory chat history, most recent last
    "notifications": [],       # pending "device" notifications for the dashboard to poll & display
}

MAX_HISTORY = 50
MAX_NOTIFICATIONS = 50
_notification_seq = 0


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


# --------------------------------------------------------------------------
# Ollama helpers
# --------------------------------------------------------------------------

def _ollama_available():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return r.ok
    except requests.RequestException:
        return False


def _model_pulled():
    """Best-effort check that llama3.2:3b is present locally."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if not r.ok:
            return False
        tags = [m.get("name", "") for m in r.json().get("models", [])]
        return any(OLLAMA_MODEL in t for t in tags)
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------
# Device notification queue
# --------------------------------------------------------------------------
# The frontend (SentinelAI dashboard) polls GET /api/notifications and renders
# each entry as an on-screen "device notification" card, the same pattern
# already used for the dashboard's own simulated alerts — this just gives the
# backend a way to push a real one (e.g. "image generated", "app launched",
# Ollama connectivity issues) instead of the frontend only ever generating
# them on its own.

def push_notification(title, body, icon="🔔", kind="info"):
    global _notification_seq
    _notification_seq += 1
    note = {
        "id": _notification_seq,
        "title": title,
        "body": body,
        "icon": icon,
        "kind": kind,          # "info" | "success" | "warning" | "error"
        "at": _now_iso(),
    }
    _state["notifications"].append(note)
    _state["notifications"] = _state["notifications"][-MAX_NOTIFICATIONS:]
    log.info("Notification queued: %s — %s", title, body)
    return note


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("jarvis_new_1.html")


@app.get("/health")
def health():
    ollama_up = _ollama_available()
    return jsonify({
        "status": "ok",
        "service": "sentinelai-ollama-bridge",
        "server_mode": _state["server_mode"],
        "ollama_reachable": ollama_up,
        "model": OLLAMA_MODEL,
        "model_available": _model_pulled() if ollama_up else False,
    })


@app.get("/api/server-mode")
def get_server_mode():
    return jsonify({
        "server_mode": _state["server_mode"],
        "last_toggled_at": _state["last_toggled_at"],
    })


@app.post("/api/server-mode")
def set_server_mode():
    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify({"error": "body must be {'enabled': true|false}"}), 400

    _state["server_mode"] = enabled
    _state["last_toggled_at"] = _now_iso()
    log.info("Server mode set to %s", enabled)
    return jsonify({
        "server_mode": _state["server_mode"],
        "last_toggled_at": _state["last_toggled_at"],
    })


@app.post("/api/chat")
def chat():
    """Non-streaming chat. Body: {"message": "...", "history": [...]?}"""
    if not _state["server_mode"]:
        return jsonify({"error": "AI assistant server mode is OFF"}), 503

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "'message' is required"}), 400

    messages = (payload.get("history") or []) + [{"role": "user", "content": message}]

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Ollama request failed: %s", exc)
        return jsonify({"error": "Could not reach Ollama", "detail": str(exc)}), 502

    data = resp.json()
    reply = data.get("message", {}).get("content", "").strip()

    _state["conversation_log"].append({"role": "user", "content": message, "at": _now_iso()})
    _state["conversation_log"].append({"role": "assistant", "content": reply, "at": _now_iso()})
    _state["conversation_log"] = _state["conversation_log"][-MAX_HISTORY:]

    return jsonify({"reply": reply, "model": OLLAMA_MODEL, "at": _now_iso()})


@app.post("/api/chat/stream")
def chat_stream():
    """Streaming chat — plain-text chunks as they arrive from Ollama."""
    if not _state["server_mode"]:
        return jsonify({"error": "AI assistant server mode is OFF"}), 503

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "'message' is required"}), 400

    messages = (payload.get("history") or []) + [{"role": "user", "content": message}]

    def generate():
        try:
            with requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
                timeout=REQUEST_TIMEOUT,
                stream=True,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except requests.RequestException as exc:
            yield f"\n[error contacting Ollama: {exc}]"

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.get("/api/conversation")
def get_conversation():
    return jsonify({"history": _state["conversation_log"]})


@app.delete("/api/conversation")
def clear_conversation():
    _state["conversation_log"] = []
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# App launcher — allow-listed only, see ALLOWED_APPS above
# --------------------------------------------------------------------------

@app.get("/api/apps")
def list_apps():
    """Lists the apps this service is permitted to open. The frontend should
    build its "open app" UI from this list rather than letting the person
    type a free-form command."""
    return jsonify({"apps": sorted(ALLOWED_APPS.keys())})


def _launch_app(name):
    """Shared launch logic used by both /api/apps/open and /api/voice-command.
    Only ever acts on a name already present in ALLOWED_APPS — callers must
    check that before calling this. Runs with this process's own (non-root)
    privileges; never elevates."""
    entry = ALLOWED_APPS[name]
    kind = entry.get("kind")
    try:
        if kind == "url":
            opened = webbrowser.open(entry["target"])
            if not opened:
                raise OSError("no browser handler available on this host")
        elif kind == "exec":
            subprocess.Popen(entry["cmd"])
        else:
            raise ValueError(f"unknown allow-list entry kind: {kind!r}")
    except (OSError, ValueError, KeyError) as exc:
        log.error("Failed to launch '%s': %s", name, exc)
        push_notification(f"Failed to open {name}", str(exc), icon="⚠️", kind="error")
        return jsonify({"error": f"Failed to launch '{name}'", "detail": str(exc)}), 500

    log.info("Launched app '%s' -> %s", name, entry)
    push_notification(f"{name} opened", "Launched on the host device.", icon="🖥️", kind="success")
    return jsonify({"ok": True, "name": name, "at": _now_iso()})


@app.post("/api/apps/open")
def open_app():
    """Body: {"name": "<one of /api/apps>"}. Only ever triggers the exact,
    pre-approved action configured for that name in ALLOWED_APPS — this
    endpoint does not accept or execute a caller-supplied command/path/URL.

    Gated on server_mode: this is what makes the assistant's "kill switch"
    real. When server_mode is OFF, the AI assistant has no ability to open
    anything — flipping it back on is what restores that ability, mirroring
    /api/chat's own server_mode check above."""
    if not _state["server_mode"]:
        return jsonify({"error": "AI assistant server mode is OFF"}), 503

    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()

    if not name:
        return jsonify({"error": "'name' is required"}), 400
    if name not in ALLOWED_APPS:
        return jsonify({
            "error": f"'{name}' is not in the allow-list",
            "allowed": sorted(ALLOWED_APPS.keys()),
        }), 403

    return _launch_app(name)


@app.post("/api/voice-command")
def voice_command():
    """Body: {"transcript": "open whatsapp"}. The frontend does speech-to-text
    itself (e.g. the browser's built-in SpeechRecognition) and posts the
    resulting text here. This endpoint does NOT execute the transcript as a
    command — it only checks whether an allow-listed app name appears in it,
    and if so triggers that exact pre-approved action. Free-form transcript
    text is never passed to a shell or subprocess."""
    if not _state["server_mode"]:
        return jsonify({"error": "AI assistant server mode is OFF"}), 503

    payload = request.get_json(silent=True) or {}
    transcript = (payload.get("transcript") or "").lower().strip()
    if not transcript:
        return jsonify({"error": "'transcript' is required"}), 400

    matched = next((name for name in ALLOWED_APPS if name in transcript), None)
    if not matched:
        return jsonify({
            "error": "No known app name found in transcript",
            "allowed": sorted(ALLOWED_APPS.keys()),
        }), 404

    return _launch_app(matched)


# --------------------------------------------------------------------------
# Image generation
# --------------------------------------------------------------------------
# Talks to a local image-gen backend that speaks the Automatic1111 / SD WebUI
# style API (POST /sdapi/v1/txt2img -> {"images": ["<base64>", ...]}). Point
# IMAGE_GEN_HOST at whatever local server you're running; swap the request
# body/parsing below if yours uses a different API shape (e.g. ComfyUI).

@app.post("/api/generate-image")
def generate_image():
    """Body: {"prompt": "...", "negative_prompt": "..."?, "width": 512?, "height": 512?, "steps": 20?}"""
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "'prompt' is required"}), 400

    body = {
        "prompt": prompt,
        "negative_prompt": payload.get("negative_prompt", ""),
        "width": int(payload.get("width", 512)),
        "height": int(payload.get("height", 512)),
        "steps": int(payload.get("steps", 20)),
    }

    try:
        resp = requests.post(
            f"{IMAGE_GEN_HOST}/sdapi/v1/txt2img",
            json=body,
            timeout=IMAGE_GEN_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Image-gen backend request failed: %s", exc)
        push_notification("Image generation failed", "Could not reach the image-gen backend.", icon="⚠️", kind="error")
        return jsonify({"error": "Could not reach image generation backend", "detail": str(exc)}), 502

    data = resp.json()
    images = data.get("images") or []
    if not images:
        push_notification("Image generation failed", "Backend returned no image.", icon="⚠️", kind="error")
        return jsonify({"error": "No image returned by backend"}), 502

    # Save the first image to disk and return both a base64 payload and a path,
    # so the frontend can either render it inline or fetch it as a file.
    filename = f"img_{int(datetime.utcnow().timestamp()*1000)}.png"
    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(images[0]))
    except (OSError, ValueError) as exc:
        log.error("Failed to save generated image: %s", exc)
        filepath = None

    push_notification("Image generated", prompt[:80], icon="🖼️", kind="success")
    return jsonify({
        "image_base64": images[0],
        "file": filepath,
        "prompt": prompt,
        "at": _now_iso(),
    })


@app.get("/api/generated-images")
def list_generated_images():
    """Lists filenames of previously generated images in IMAGE_OUTPUT_DIR."""
    try:
        files = sorted(os.listdir(IMAGE_OUTPUT_DIR), reverse=True)
    except OSError:
        files = []
    return jsonify({"images": files})


# --------------------------------------------------------------------------
# Device notifications — pushed by the backend, polled by the dashboard
# --------------------------------------------------------------------------

@app.get("/api/notifications")
def get_notifications():
    """Optional query param ?since=<id> returns only notifications newer
    than that id, so the frontend can poll without re-rendering everything."""
    since = request.args.get("since", type=int, default=0)
    pending = [n for n in _state["notifications"] if n["id"] > since]
    return jsonify({"notifications": pending, "latest_id": _notification_seq})


@app.post("/api/notifications")
def create_notification():
    """Lets other parts of the backend (or an operator/testing tool) push a
    notification manually. Body: {"title": "...", "body": "...", "icon": "🔔"?, "kind": "info"?}"""
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    body = (payload.get("body") or "").strip()
    if not title:
        return jsonify({"error": "'title' is required"}), 400
    note = push_notification(title, body, icon=payload.get("icon", "🔔"), kind=payload.get("kind", "info"))
    return jsonify(note), 201


@app.delete("/api/notifications")
def clear_notifications():
    _state["notifications"] = []
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Vision — face detection
# --------------------------------------------------------------------------

@app.post("/api/vision/detect-faces")
def api_detect_faces():
    """Body: {"image_base64": "..."} (data: URL prefix is fine). Returns
    bounding boxes. Not gated on server_mode — it's a passive vision check,
    not an action — but you can add that guard if you want it off by default."""
    payload = request.get_json(silent=True) or {}
    image_b64 = (payload.get("image_base64") or "").strip()
    if not image_b64:
        return jsonify({"error": "'image_base64' is required"}), 400

    result = detect_faces_b64(image_b64)
    if "error" in result:
        return jsonify(result), 400

    push_notification(
        f"{result['face_count']} face(s) detected", "", icon="🙂",
        kind="info" if result["face_count"] else "info",
    )
    return jsonify(result)


# --------------------------------------------------------------------------
# Skills — automated skill drafting + "self-thinking" reflection
# --------------------------------------------------------------------------
# SECURITY: generate_skill() only WRITES a .py draft to disk for you to
# review — nothing here imports or executes model-generated code
# automatically. Gated on server_mode like the other AI-assistant actions.

@app.post("/api/skills/generate")
def api_generate_skill():
    """Body: {"goal": "..."}. Drafts a skill file under skills/ for review."""
    if not _state["server_mode"]:
        return jsonify({"error": "AI assistant server mode is OFF"}), 503

    payload = request.get_json(silent=True) or {}
    goal = (payload.get("goal") or "").strip()
    if not goal:
        return jsonify({"error": "'goal' is required"}), 400

    result = generate_skill(goal)
    if "error" in result:
        return jsonify(result), 400

    push_notification("New skill drafted", goal[:80], icon="🧩", kind="success")
    return jsonify(result), 201


@app.get("/api/skills")
def api_list_skills():
    return jsonify({"skills": list_skills()})


@app.post("/api/skills/reflect")
def api_reflect():
    """'Self-thinking': looks at recent conversation/notifications and
    returns up to 3 plain-language suggestions. Suggests only — never
    acts on its own."""
    if not _state["server_mode"]:
        return jsonify({"error": "AI assistant server mode is OFF"}), 503

    result = reflect(_state["conversation_log"], _state["notifications"])
    if "error" in result:
        return jsonify(result), 502
    return jsonify(result)


# --------------------------------------------------------------------------
# Security — local, defensive self-audit (NOT an attack/exploit tool)
# --------------------------------------------------------------------------
# Read-only checks of THIS host's own configuration (debug mode, CORS,
# open local ports, outdated packages, file permissions, root/exec risk).
# Never touches, scans, or targets any other machine.

@app.post("/api/security/audit")
def api_security_audit():
    report = run_audit(
        flask_debug=app.debug,
        cors_origins="*",
        bind_host="0.0.0.0",
        base_dir=os.path.dirname(os.path.abspath(__file__)),
        allowed_apps=ALLOWED_APPS,
    )
    push_notification(
        "Security audit complete",
        f"{report['summary']['high']} high, {report['summary']['medium']} medium finding(s)",
        icon="🛡️",
        kind="warning" if report["summary"]["high"] else "success",
    )
    return jsonify(report)


# --------------------------------------------------------------------------
# Voice — ElevenLabs text-to-speech
# --------------------------------------------------------------------------

@app.post("/api/tts")
def api_tts():
    """Body: {"text": "...", "voice_id": "..."?, "stability": 0.5?, "similarity_boost": 0.75?}
    Returns raw MP3 bytes (audio/mpeg)."""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    try:
        audio_bytes = tts_synthesize(
            text,
            voice_id=payload.get("voice_id"),
            stability=float(payload.get("stability", 0.5)),
            similarity_boost=float(payload.get("similarity_boost", 0.75)),
        )
    except TTSError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return Response(audio_bytes, mimetype="audio/mpeg")


@app.get("/api/tts/voices")
def api_tts_voices():
    try:
        return jsonify(tts_list_voices())
    except TTSError as exc:
        return jsonify({"error": str(exc)}), exc.status_code


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting SentinelAI Ollama bridge on :%s (model=%s, ollama=%s)",
              PORT, OLLAMA_MODEL, OLLAMA_HOST)
    if not _ollama_available():
        log.warning("Ollama not reachable at %s — is `ollama serve` running?", OLLAMA_HOST)
    elif not _model_pulled():
        log.warning("Model '%s' not found locally — run: ollama pull %s", OLLAMA_MODEL, OLLAMA_MODEL)

    app.run(host="0.0.0.0", port=PORT, debug=False)
