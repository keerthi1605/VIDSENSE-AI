# VidSense AI — Architecture & Roadmap

## Problem

Traditional video search relies on title/description/tags/filename metadata and cannot
answer content-level questions like *"where does the professor explain binary search?"*.
VidSense AI indexes what's actually **said** (audio, via Whisper) and **shown**
(visual, via CLIP-style embeddings) in a video, so it can be searched semantically and
questioned via RAG (Retrieval-Augmented Generation).

## Pipeline (target, full system)

```text
Video
  |
  +--------------------+
  |                    |
  v                    v
Audio                Video Frames
  |                    |
  v                    v
Whisper              Vision Model
  |                    |
  v                    v
Transcript          Visual Embeddings
  |                    |
  +---------+----------+
            |
            v
      Multimodal Index (ChromaDB)
            |
            v
      User Question -> Query Embedding -> Retrieval
            |
            v
 Relevant Transcript + Frames -> LLM -> Answer + Evidence + Timestamp
```

## Phase roadmap

1. Video upload + Whisper transcription (timestamped)
2. Transcript chunking + text embeddings
3. Vector database + semantic search
4. RAG question answering (LLM over retrieved context)
5. Visual understanding (frame sampling + CLIP embeddings)
6. Multimodal retrieval (text + visual score fusion)
7. Timestamp-aware answers (jump-to-video)
8. Frontend (React: player + transcript + chat)
9. Database + background job processing (PostgreSQL, job states)
10. Evaluation (WER, Precision@K/Recall@K/MRR, faithfulness)
11. Optimization (reranking, hybrid search, caching, GPU/quantization)
12. Deployment (Docker Compose) + documentation

Each phase must leave the project runnable end-to-end before the next begins.

## Design principles

- **Modularity** — API / business logic (services) / AI models / storage / DB stay
  in separate layers; a service never talks to FastAPI request objects directly.
- **Timestamp preservation** — every transcript segment, chunk, and retrieval result
  carries `start_time`/`end_time` so any answer is traceable back to an exact moment
  in the source video. This must survive transcription → chunking → embedding →
  retrieval → RAG without being dropped.
- **Model abstraction** — no component hard-codes one LLM, one embedding model, or
  one vector DB. Configuration (`backend/app/core/config.py`) is the single place
  those choices live, so they can be swapped later.
- **Explainability** — every answer must be able to cite *which* transcript segment
  (with timestamps) it was based on.
- **No fake AI** — every library/technology added must solve a real problem in the
  current phase. Nothing is added "because it's popular."

## Current state (Phase 1, Step 3 — Phase 1 complete)

- FastAPI app skeleton (`backend/app/main.py`) with a `GET /health` endpoint.
- Centralized settings (`backend/app/core/config.py`) covering storage paths and
  Whisper model configuration (CPU/int8, given this dev machine's GPU is too old/
  small to accelerate inference).
- Storage directories (`videos/`, `audio/`, `frames/`, `transcripts/`, `embeddings/`)
  created under `backend/storage/`.
- `POST /api/videos/upload` — validates extension and streams the file to disk
  under a generated `video_id` (never the client-supplied filename), enforcing
  the configured size cap mid-stream. Persists a JSON metadata sidecar
  (`storage/videos/{video_id}.json`) alongside the video file.
- `POST /api/videos/transcribe` — extracts mono 16kHz WAV audio via FFmpeg
  (`app/services/audio_service.py`), then runs Faster-Whisper transcription
  (`app/services/transcription_service.py`), producing timestamped segments
  persisted to `storage/transcripts/{video_id}.json`. Synchronous (blocks the
  request) by design for Phase 1; runs on a worker thread so it doesn't stall
  the event loop. The Whisper model itself is a lazy-loaded singleton, loaded
  once per process rather than per request.
- `GET /api/videos/{video_id}/transcript` — fetches a saved transcript.
- Verified end-to-end with a real synthesized-speech test video: accurate
  transcription with correct timestamps; first transcription ~2 minutes
  (includes one-time model download + load), subsequent transcriptions in the
  same process ~2.4 seconds.
- No chunking, embeddings, vector search, or LLM logic yet — that's Phase 2.
