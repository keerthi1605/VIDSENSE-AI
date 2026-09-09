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

### Phase 2, Step 1 — Transcript chunking

- `app/services/chunking_service.py` — merges consecutive `TranscriptSegment`s
  into `Chunk`s targeting `chunk_target_words` (default 150), carrying the
  last `chunk_overlap_segments` (default 1) segments into the next chunk for
  boundary continuity. Chunk `start_time`/`end_time` are always taken
  directly from real segment timestamps — never interpolated.
- `POST /api/videos/{video_id}/chunks` — chunk a video's saved transcript
- `GET /api/videos/{video_id}/chunks` — fetch a previously computed chunk set
- Verified on real Whisper output (a two-topic synthesized lecture): with a
  lowered word target, chunk boundaries landed sensibly at the topic shift,
  and overlap was visibly preserving boundary-spanning sentences across
  adjacent chunks.

### Phase 2, Step 2 — Text embeddings (Phase 2 complete)

- `app/services/embedding_service.py` — Sentence-Transformers
  (`all-MiniLM-L6-v2`, 384-dim, CPU) as a lazy-loaded singleton, same pattern
  as the Whisper model. `embed_texts()` produces L2-normalized vectors;
  `cosine_similarity()` is a small standalone utility.
- Vectors persisted as compact binary `.npy` (not JSON — ~230KB vs. several x
  larger for 150 chunks as a float JSON array), with an `EmbeddingSet` JSON
  metadata sidecar (model name, dimension, chunk_ids in row order). Raw
  vectors are deliberately never returned over the API.
- `POST/GET /api/videos/{video_id}/embeddings`
- Verified two ways:
  1. A real-model unit test: two paraphrased sentences score higher cosine
     similarity than an unrelated one (not just trusting the library).
  2. A manual end-to-end script: chunked + embedded a real two-topic
     transcript, then hand-computed cosine similarity between two queries
     and all chunks. Both queries correctly ranked their matching-topic
     chunks far above the unrelated ones (e.g. the deadlock-conditions query
     scored 0.85 against the deadlock-conditions chunk vs. 0.08 against the
     binary-search chunk) — this is the same mechanism Phase 3 formalizes
     with ChromaDB instead of a hand-rolled loop.
## Phase 3 — Vector Database + Semantic Search (complete)

- `app/services/vector_store_service.py` — the ONLY module that imports
  chromadb, keeping the vector DB swappable (e.g. to FAISS) without touching
  anything above it. Uses a `PersistentClient` (local folder, no server
  process), a collection explicitly configured for `hnsw:space: cosine`
  (Chroma defaults to squared-L2, which would silently rank differently than
  the cosine similarity validated since Phase 2), and `upsert` (not `add`)
  for idempotent re-indexing. Telemetry disabled.
- `app/services/retrieval_service.py` — domain-level `index_video()` (push
  chunks+embeddings into the vector store, after verifying chunk_ids and
  embedding chunk_ids actually align) and `search()` (embed query -> vector
  store query -> shaped `SearchResult`s, `score = 1 - distance` so "higher is
  better" everywhere).
- `POST /api/videos/{video_id}/index` — index a video's chunks+embeddings
- `GET`/`POST /api/search` — semantic search, optionally filtered to one
  `video_id`, returning `{video_id, chunk_id, text, start_time, end_time,
  score}` per result. No LLM involved — every result is directly traceable
  to an exact place in a video, nothing generated.
- Verified on real Whisper output through the full HTTP pipeline (upload ->
  transcribe -> chunk -> embed -> index -> search) on a two-topic lecture:
  a deadlock-conditions query correctly ranked the deadlock-conditions chunk
  first (score 0.80) over the binary-search chunk (0.50); a binary-search
  query correctly ranked the binary-search chunks first; the `video_id`
  filter correctly restricted results to one video, including returning an
  empty (not erroring) result set for a video with no indexed chunks.
- No LLM / answer-generation logic yet — that's Phase 4.
