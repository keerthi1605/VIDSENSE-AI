# VidSense AI

Multi-modal semantic video search and question answering using RAG (Retrieval-Augmented Generation).

Upload a long video (lecture, tutorial, meeting) and ask natural-language questions about it —
the system understands both the **spoken audio** and the **visual content**, retrieves the
relevant moment, and answers with a timestamp you can jump to.

## Project status

Being built incrementally, phase by phase. Each phase is left in a stable, working state
before the next begins — see [`docs/architecture.md`](docs/architecture.md) for the full roadmap.

- [x] Phase 1 — Video upload + Whisper transcription — **complete**
- [x] Phase 2 — Transcript chunking + text embeddings — **complete**
- [x] Phase 3 — Vector database + semantic search — **complete**
- [x] Phase 4 — RAG question answering — **complete**
- [ ] Phase 5 — Visual understanding (frame extraction + CLIP) — **Step 1 (frame extraction) done, Step 2 (CLIP embeddings) next**
- [ ] Phase 6 — Multimodal retrieval
- [ ] Phase 7 — Timestamp-aware answers
- [ ] Phase 8 — Frontend
- [ ] Phase 9 — Database + background job processing
- [ ] Phase 10 — Evaluation
- [ ] Phase 11 — Optimization
- [ ] Phase 12 — Deployment

## Tech stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic
- **Video/audio:** FFmpeg, OpenCV
- **Speech-to-text:** Faster-Whisper (CPU/int8 — see hardware note below)
- **Embeddings:** Sentence-Transformers (text), CLIP (visual, added Phase 5)
- **Vector DB:** ChromaDB
- **LLM:** provider-agnostic by design (`app/services/llm_service.py`); currently
  Ollama running `qwen2.5:3b` locally (CPU-forced — see Hardware note)
- **Frontend:** React (added Phase 8)

## Hardware note

Developed on a machine with an old/low-VRAM GPU (2GB, pre-CUDA-11 driver), so all AI
inference defaults to **CPU**, using smaller/quantized models. This is reflected in
`backend/app/core/config.py` and `backend/.env.example`. Swapping to GPU on a better
machine is a config change, not a code change.

**Important — Ollama on this GPU:** Ollama's default backend probes for a usable GPU
via Vulkan and finds the GeForce MX230, but that GPU **does not support 16-bit
storage**, which crashes `llama-server` (`ggml_vulkan: device Vulkan0 does not support
16-bit storage`) on every generation request. The fix is forcing Ollama onto its CPU
backend:

```powershell
[System.Environment]::SetEnvironmentVariable("OLLAMA_LLM_LIBRARY", "cpu", "User")
```

Set this once, then restart the Ollama app/service (log out/in, or restart the
process) so it takes effect. Without it, `/api/chat` will return a 503 the moment
Ollama actually tries to generate a response.

## Setting up Ollama (Phase 4)

```powershell
winget install --id Ollama.Ollama -e
# See the GPU note above BEFORE first use on this hardware.
ollama pull qwen2.5:3b
```

Ollama runs as a background service listening on `http://localhost:11434` (the
default `OLLAMA_BASE_URL` in `.env.example`). Confirm it's up with
`ollama list` (should show `qwen2.5:3b`) or `curl http://localhost:11434/api/tags`.

## Running the backend (Phase 1)

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# CPU-only torch build first -- much smaller download than the default
# (which bundles an unused CUDA runtime); this project runs all AI
# inference on CPU by design (see Hardware note above).
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Then check:

```bash
curl http://127.0.0.1:8000/health
```

## API (Phases 1–4): upload to answered question

```bash
# 1. Upload a video, note the returned video_id
curl -X POST http://127.0.0.1:8000/api/videos/upload -F "file=@lecture.mp4"

# 2. Transcribe it (first call downloads+loads the Whisper model, ~1-2 min;
#    later calls in the same running process take a few seconds)
curl -X POST http://127.0.0.1:8000/api/videos/transcribe \
  -H "Content-Type: application/json" \
  -d '{"video_id": "<video_id from step 1>"}'

# 3. Fetch the transcript any time after
curl http://127.0.0.1:8000/api/videos/<video_id>/transcript

# 4. Chunk the transcript into embedding-sized pieces
curl -X POST http://127.0.0.1:8000/api/videos/<video_id>/chunks

# 5. Fetch the chunk set any time after
curl http://127.0.0.1:8000/api/videos/<video_id>/chunks

# 6. Generate embeddings for the chunks
curl -X POST http://127.0.0.1:8000/api/videos/<video_id>/embeddings

# 7. Fetch embedding metadata (model, dimension, chunk count -- not the
#    raw vectors themselves; those live in storage/embeddings/*.npy)
curl http://127.0.0.1:8000/api/videos/<video_id>/embeddings

# 8. Push the chunks + embeddings into the vector database
curl -X POST http://127.0.0.1:8000/api/videos/<video_id>/index

# 9. Semantic search -- across all indexed videos, or one via video_id
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "explain deadlock prevention", "top_k": 5}'

curl -G http://127.0.0.1:8000/api/search --data-urlencode "query=binary search"

# 10. Ask a question -- retrieval + LLM answer + exact-timestamp sources
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the four necessary conditions for deadlock?"}'

# 11. Extract representative frames (fixed interval + near-duplicate skip)
curl -X POST http://127.0.0.1:8000/api/videos/<video_id>/frames

# 12. Fetch the frame manifest, or view one frame's actual image
curl http://127.0.0.1:8000/api/videos/<video_id>/frames
curl http://127.0.0.1:8000/api/videos/<video_id>/frames/<frame_id>/image -o frame.jpg
```

## Repository structure

```text
vidsense-ai/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app entrypoint
│   │   ├── api/              # route handlers
│   │   ├── services/         # business logic (video, audio, transcription, ...)
│   │   ├── models/           # Pydantic schemas
│   │   ├── core/             # config, logging
│   │   └── utils/
│   ├── storage/               # videos / audio / frames / transcripts / embeddings
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── docs/
│   └── architecture.md
└── README.md
```

(`frontend/` and `docker-compose.yml` are added in later phases — not created early to
avoid unused scaffolding.)
