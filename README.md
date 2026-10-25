# SentinelAI — everything merged into ollama_service2.py

```
sentinelai/
├── requirements.txt
├── .env.example                  # copy -> .env, fill in real keys
├── README.md
└── app/
    ├── ollama_service2.py         # YOUR original file, now with every new route added in-place
    ├── modules/
    │   ├── face_detection.py      # OpenCV face detection
    │   ├── skills_engine.py       # skill-drafting + "self-thinking" reflection
    │   ├── elevenlabs_tts.py      # ElevenLabs TTS, plain functions (no FastAPI)
    │   └── security_audit.py      # local, defensive vulnerability self-audit
    ├── templates/
    │   └── jarvis.html            # keep your existing dashboard HTML here
    ├── skills/                    # AUTO-GENERATED skill drafts land here
    │                               #   (never auto-executed -- review first)
    ├── reports/                   # security audit reports (.md + .json) land here
    └── generated_images/          # existing image-gen output folder
```

Everything now lives in **one Flask app, one process, one port** -- exactly
your original `ollama_service2.py`, just with four new route groups added
at the bottom of the routes section, right before the entrypoint. Nothing
in your existing routes was changed.

## Run it

```bash
cd sentinelai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add your ELEVENLABS_API_KEY

cd app
ollama serve &                 # terminal 1 (or already running)
python ollama_service2.py      # terminal 2 -- same as before, now on :5001
```

## New endpoints (all on the same :5001 service)

| Method | Path                       | Gated on server_mode? | What it does |
|--------|----------------------------|:---:|---|
| POST   | `/api/vision/detect-faces` | no  | `{"image_base64": "..."}` -> bounding boxes |
| POST   | `/api/skills/generate`     | yes | `{"goal": "..."}` -> drafts a skill file in `skills/` for review |
| GET    | `/api/skills`              | no  | Lists drafted skill files |
| POST   | `/api/skills/reflect`      | yes | "Self-thinking": 1-3 plain-language suggestions from recent activity |
| POST   | `/api/security/audit`      | no  | Runs the local, read-only security self-audit, writes a report |
| POST   | `/api/tts`                 | no  | `{"text": "..."}` -> MP3 bytes (ElevenLabs) |
| GET    | `/api/tts/voices`          | no  | Lists available ElevenLabs voices |

(`server_mode`-gated routes mirror how `/api/chat` and `/api/apps/open`
already behave -- flip it on via `POST /api/server-mode`.)

## Design notes -- why it's built this way

- **Skills are never auto-executed.** `generate_skill()` only writes a `.py`
  file to `skills/` with the model's draft. You import/run it yourself after
  reading it -- an assistant that silently executes its own generated code
  on your machine is a remote-code-execution risk.
- **"Self-thinking" is a suggestion queue**, not an autonomous actor. It reads
  recent chat/notifications and proposes next steps in plain text; nothing
  fires on its own.
- **The security/"hacking" mode is a defensive local audit only** -- it checks
  *this* host's own configuration (debug mode, CORS, open ports, outdated
  packages, file permissions, root/exec risk) and writes a report. It does
  not scan, probe, or attack any other machine, and generates no exploit
  code.
- **ElevenLabs key stays in `.env`**, same pattern as `OLLAMA_HOST` etc.

## Still to wire up yourself

- Face detection needs a webcam frame from the browser -- capture a `<video>`
  frame to a `<canvas>`, `canvas.toDataURL()`, POST that as `image_base64`.
- `jarvis.html` isn't touched here; add buttons/calls for the new endpoints
  when you're ready, or ask and I'll build that next.
