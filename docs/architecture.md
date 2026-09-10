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

## Phase 4 — RAG Question Answering (complete)

- `app/services/llm_service.py` — the LLM provider abstraction: an `LLMProvider`
  ABC with one method, `generate(prompt) -> str`. `OllamaProvider` is the only
  implementation today, calling a local Ollama server's `/api/generate` HTTP
  endpoint. `get_llm_provider()` is the single factory function everything else
  calls — swapping to a cloud provider later means adding one class and one
  branch here, not touching rag_service or the API layer.
- `app/services/rag_service.py` — owns the prompt template and the RAG
  orchestration: `search()` for context -> build a prompt from the retrieved
  chunks -> `generate()` -> attach sources. **Sources are built directly from
  the retrieved `SearchResult`s, never parsed out of the LLM's own text** — a
  3B local model asked to also emit structured citations (e.g. JSON) is
  unreliable, so the model's only job is the natural-language answer; exact
  timestamps come from data already trusted, not from hoping the model gets
  a citation format right.
- `has_sufficient_context` is a deterministic check (did retrieval return any
  chunks at all), not a parse of whether the LLM's own answer claims it could
  or couldn't answer. The prompt instructs the model to say so itself when
  the excerpts are insufficient — a real behavior, just not yet fed back into
  this boolean, since reliably parsing free text from a small local model is
  its own source of bugs.
- `POST /api/chat` — `{query, top_k?, video_id?}` -> `{query, answer, sources,
  has_sufficient_context}`. Returns 503 (not 500) when Ollama is unreachable
  or times out, with an actionable message.
- Model choice: `qwen2.5:3b` via Ollama, chosen after checking this machine's
  actual hardware (8GB total RAM, no usable GPU) — a 3B instruction-tuned
  model is close to the practical ceiling for reliable CPU inference here.
- **Hardware-specific fix required**: Ollama's default backend probes for a
  GPU via Vulkan and finds the GeForce MX230, but that GPU doesn't support
  16-bit storage, which crashes `llama-server` on every generation call.
  Fixed by setting `OLLAMA_LLM_LIBRARY=cpu` (persisted at the user
  environment level) to force Ollama onto its CPU backend. Documented in
  the README since it's a one-time, easy-to-miss setup step for this exact
  hardware.
- Verified end-to-end on a live server with a real two-topic lecture (indexed
  as 6 chunks): a deadlock-conditions question correctly retrieved and cited
  the deadlock chunk (score 0.79) and produced an accurate, non-hallucinated
  answer; a binary-search question did the same for the binary-search chunk
  (score 0.78) while ranking the deadlock chunks near the bottom (0.09-0.17);
  a query against a nonexistent `video_id` correctly short-circuited to the
  deterministic "not enough information" response without calling the LLM;
  killing the Ollama process and retrying correctly returned a 503 with a
  clear message, then recovered once Ollama was restarted.
- No conversation memory/multi-turn chat — out of scope for this phase's
  "retrieval-grounded single-turn answering" goal; a real feature to add
  later, not built here to avoid scope creep.
