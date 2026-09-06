# VidSense AI

Multi-modal semantic video search and question answering using RAG (Retrieval-Augmented Generation).

Upload a long video (lecture, tutorial, meeting) and ask natural-language questions about it —
the system understands both the **spoken audio** and the **visual content**, retrieves the
relevant moment, and answers with a timestamp you can jump to.

## Project status

Being built incrementally, phase by phase. Each phase is left in a stable, working state
before the next begins — see [`docs/architecture.md`](docs/architecture.md) for the full roadmap.

- [x] Phase 1 — Video upload + Whisper transcription — **Step 2 (video upload) done, Step 3 (transcription) next**
- [ ] Phase 2 — Transcript chunking + text embeddings
- [ ] Phase 3 — Vector database + semantic search
- [ ] Phase 4 — RAG question answering
- [ ] Phase 5 — Visual understanding (frame extraction + CLIP)
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
- **LLM:** provider-agnostic by design (local via Ollama, or a swappable API)
- **Frontend:** React (added Phase 8)

## Hardware note

Developed on a machine with an old/low-VRAM GPU (2GB, pre-CUDA-11 driver), so all AI
inference defaults to **CPU**, using smaller/quantized models. This is reflected in
`backend/app/core/config.py` and `backend/.env.example`. Swapping to GPU on a better
machine is a config change, not a code change.

## Running the backend (Phase 1)

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Then check:

```bash
curl http://127.0.0.1:8000/health
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
