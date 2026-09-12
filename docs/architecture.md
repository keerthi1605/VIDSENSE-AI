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

## Phase 5, Step 1 — Frame Extraction

- `app/services/frame_service.py` — samples frames via OpenCV, not FFmpeg
  (comparing pixel data to detect duplicates needs actual arrays in hand,
  not a second read-back pass over files FFmpeg already wrote). Two-part
  sampling strategy:
  1. Fixed-interval cadence (`frame_sample_interval_seconds`, default 5s)
     bounds how often a frame is even considered.
  2. Near-duplicate skip: each candidate is compared to the last SAVED
     frame via mean absolute pixel difference on a small downsampled
     grayscale copy; below `frame_diff_threshold` (default 10.0, a
     starting heuristic) it's treated as the same slide/scene and skipped.
  Uses `cap.grab()`/`cap.retrieve()` (not `cap.set(POS_FRAMES, ...)` seeking)
  for sequential, codec-portable timestamps without decoding skipped frames.
- `Frame`/`FrameSet` schemas — one manifest JSON per video
  (`storage/frames/{video_id}/manifest.json`), images alongside it as
  `frame_0000.jpg`, etc. Deliberately deviates from the master prompt's
  example JSON (which shows `embedding` inline) — mirrors Phase 2's
  established pattern of keeping vectors in a separate binary file, added
  in Step 2 once frames themselves are verified.
- `POST/GET /api/videos/{video_id}/frames`, plus
  `GET /api/videos/{video_id}/frames/{frame_id}/image` to serve the actual
  JPEG (useful for a frontend preview or manual inspection).
- Verified two ways:
  1. Synthetic videos with real, controlled pixel content: a static-color
     video produces exactly 1 saved frame (everything else correctly
     identified as duplicate); a two-scene video (color change at t=5s)
     produces exactly 2 frames — proof the duplicate check discriminates
     real content changes from a static scene, not just an assumption.
  2. Live end-to-end: a 3-segment color-change video (18s, cuts at 6s/12s)
     through the real upload -> extract-frames API produced exactly 3
     frames at t=0/10/15s (the t=5 candidate correctly skipped as a
     duplicate of t=0); downloaded and visually confirmed the served JPEG
     matched the expected color for its timestamp.
- No visual embeddings yet (CLIP) — that's Step 2. Frames aren't searchable
  yet, just extracted and inspectable.

## Phase 5, Step 2 — CLIP Visual Embeddings

- `app/services/vision_service.py` — a separate file from
  `embedding_service.py` on purpose (matches the master-prompt architecture
  and keeps the two modalities' lazy-singleton models independent).
  `clip-ViT-B-32` via sentence-transformers (already a dependency) — CLIP
  is the only real option here, not one choice among several: it's the
  only model that embeds images and text into the SAME vector space,
  which is what makes "search frames by a text query" possible at all.
  This is a completely separate 512-dim space from `embedding_model_name`'s
  384-dim text-only space — the two are never compared to each other.
- `embed_text_for_visual()` embeds a query into CLIP's space (for Phase 6's
  visual search) using the exact same model/method as `embed_images()` —
  included now, not deferred, because it's what makes the joint-space
  property testable at all.
- `FrameEmbeddingSet` schema + `.npy`/JSON sidecar storage
  (`storage/embeddings/{video_id}_frames.{npy,json}`), same separation
  principle as `EmbeddingSet`: vectors never returned raw over the API.
- `POST/GET /api/videos/{video_id}/frame-embeddings`
- Verified two ways:
  1. A real-model test: two solid-color images (red, blue) and their
     matching/mismatching text descriptions — the red image scores higher
     against "a solid red image" than "a solid blue image," and vice
     versa. Proof the joint space actually holds, not just an assumption
     about what CLIP should do.
  2. Live end-to-end + a manual "poor man's visual search": embedded a
     real 3-segment color video's frames via the live API, then queried
     CLIP's text encoder with "a dark navy blue background" / "a dark
     green background" / "a dark maroon red background" against the
     saved vectors — all three correctly ranked their matching-color
     frame first (e.g. 0.33 vs ~0.25-0.26 for the other two), the same
     mechanism Phase 6 will use for real multimodal retrieval.
- **Operational note**: HuggingFace's `xet` transfer backend was
  unreliable on this network during the ~600MB CLIP download (bandwidth
  collapsed to a few KB/s with repeated retries after an initially normal
  rate). Fixed by setting `HF_HUB_DISABLE_XET=1` to force the plain HTTP
  downloader, which completed reliably. Documented in the README since
  it'll recur for any large model download on this connection.
- No vector-store indexing yet for frames — chunks index into ChromaDB
  (Phase 3), but frame vectors are still file-only. That's Step 3, which
  will also need to decide how a "video_frames" collection coexists with
  the existing "video_chunks" one before Phase 6 can fuse both searches.
