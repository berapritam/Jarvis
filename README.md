# Jarvis

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





cd sentinelai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add your ELEVENLABS_API_KEY

cd app
ollama serve &                 # terminal 1 (or already running)
python ollama_service2.py      # terminal 2 -- same as before, now on :5001
