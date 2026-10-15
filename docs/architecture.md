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

### Phase 1, Step 4 — Ingest a video from a URL (YouTube, etc.)

- `app/services/video_service.save_video_from_url()` — downloads via `yt-dlp` and
  stores the result **exactly like a direct upload** (same `VideoMetadata` shape,
  same `storage/videos/{video_id}.mp4` layout) so every downstream endpoint
  (transcribe, chunks, embeddings, frames, ...) works unchanged regardless of how
  the video arrived. `source_url` is the only new field, kept purely for
  provenance.
- Two-phase by design: **probe** the video's metadata first (`extract_info(...,
  download=False)`) and reject a too-long video or a live stream *before*
  spending any bandwidth/disk — only then actually download.
- Resolution capped at 720p in the format selector (`yt_dlp_format`) — this
  machine is CPU-only and storage-conscious by design; no reason to pull 4K.
  `merge_output_format` + an `FFmpegVideoConvertor` postprocessor guarantee a
  consistent `.mp4` regardless of the source container.
- Same "never trust an external name for a path" discipline as the direct
  upload: the file is saved under the generated `video_id`, never the
  remote-provided title (which becomes a display-only field, `original_filename`).
- `POST /api/videos/from-url` — `{"url": "..."}` → the same `VideoMetadata`
  response shape as `/upload`.
- **Real bug caught and fixed during verification**: `ffmpeg_location` was
  initially passed the bare command name (`"ffmpeg"`), which yt-dlp does NOT
  resolve via PATH the way a shell would — it failed with "ffmpeg is not
  installed" despite ffmpeg being on PATH and working everywhere else in this
  project. Fixed by resolving it explicitly via `shutil.which()` first.
- Verified two ways:
  1. Mocked `yt_dlp.YoutubeDL` entirely (`test_video_from_url.py`) — no network
     in the automated suite. Covers: successful download, duration-limit
     rejection, live-stream rejection, probe failure (400), download failure
     with partial-file cleanup (502), missing-output-file (500), and that a
     hostile/weird remote title never influences the stored path.
  2. **Real live download**: used yt-dlp's own search syntax
     (`ytsearch1:Big Buck Bunny trailer official Blender Foundation`, a
     Creative-Commons-licensed official Blender Foundation short) to avoid
     hardcoding a URL that might rot. Downloaded a real 81.8MB, 720p, ~10.5-minute
     AV1 video, then ran it through the real `/transcribe` endpoint — the
     result was Whisper hallucinating "I'm sorry" at each ~30s window boundary
     (the video has no dialogue, only music/sound effects), a textbook
     real-world instance of the exact hallucination-on-non-speech-audio failure
     mode documented in `docs/concepts.md`'s Phase 1 section. The pipeline
     integration itself was proven correct: identical request/response shapes
     to a direct upload, no special-casing needed anywhere downstream.

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

## Phase 5, Step 3 — Indexing Frames into the Vector Store (Phase 5 complete)

- `vector_store_service.py` **refactored to be collection-agnostic**:
  `get_collection()`, `upsert()` (renamed from `upsert_chunks`), `query()`,
  and `count()` all take an explicit `collection_name` now, and `query()`
  returns generic `{id, document, metadata, score}` dicts instead of
  chunk-specific fields. This module has no idea what a "chunk" or "frame"
  is — that interpretation moved up into `retrieval_service.py`, split
  into a text section and a visual section.
- Frame vectors (512-dim, CLIP) live in a **separate** collection
  (`video_frames`) from text chunks (384-dim, MiniLM, `video_chunks`) —
  Chroma collections are single-dimension, and the two spaces are
  non-comparable anyway (see `docs/concepts.md`).
- `retrieval_service.index_video_frames()` / `search_frames()` — mirror
  `index_video()`/`search()` exactly, with one critical difference:
  `search_frames()` embeds the query via `vision_service.embed_text_for_visual()`
  (CLIP's text encoder), never `embedding_service.embed_texts()` (MiniLM) —
  covered by a dedicated test asserting MiniLM is never called.
- `POST /api/videos/{video_id}/index-frames` — a separate endpoint from
  `/index` (text), since a video can have one pipeline without the other.
- `GET`/`POST /api/search/visual` — a standalone, unfused visual search
  endpoint, deliberately built the same way Phase 3's `/api/search` was
  built before Phase 4 added the LLM: prove each retrieval channel works
  independently before Phase 6 fuses their two ranked lists.
- Verified three ways:
  1. Real-Chroma integration test (`test_visual_search_api.py`, mirrors
     `test_search_api.py`'s structure): a fake-but-deterministic CLIP
     model, real indexing + search through the actual HTTP endpoints,
     plus an explicit test that indexing frames leaves the text
     collection's count at zero (proof the two collections don't
     cross-contaminate).
  2. Live end-to-end with the REAL CLIP model: a real 3-segment
     color-change video run through upload → frames → frame-embeddings →
     index-frames → `/api/search/visual`. All three text queries ("navy
     blue" / "dark green" / "maroon red") correctly ranked their
     matching-color frame first via the actual production API path (not
     just the earlier manual script) — e.g. 0.31 vs ~0.25 for navy,
     0.33 vs ~0.24-0.26 for maroon.
  3. Confirmed `/api/search` (text) returns empty after only frames were
     indexed — the two collections are genuinely independent, not just
     asserted to be.

Phase 5 (frame extraction, CLIP embeddings, vector-store indexing) is now
complete. The visual pipeline is a full parallel of the text pipeline
(Phases 2+3), ready for Phase 6 to fuse both searches into one ranked list.

## Phase 6 — Multimodal Retrieval (text + visual fusion) (complete)

- `retrieval_service.search_multimodal()` — fuses the two existing, independent
  searches (MiniLM over transcript chunks, CLIP over frames) into ONE ranked
  list of moments. Exposed as `POST`/`GET /api/search/multimodal`.
- **Algorithm**: (1) pull `multimodal_candidate_k` (10) candidates from each
  modality; (2) **min-max normalize each modality's scores within its own list**
  — MiniLM matches score ~0.7–0.85 while CLIP matches score only ~0.25–0.35, so
  an unnormalized sum would let text win regardless of weights; (3) **align by
  time** — a frame whose timestamp lies inside a chunk's `[start, end]` (same
  video) is evidence for the same moment and merges into one `both` result
  (chunk text + best-matching frame); unmatched chunks stay `text`, unmatched
  frames stay `visual` (`start == end ==` frame timestamp, so the frontend always
  has a valid range to seek to); (4) `final = text_weight·text + visual_weight·visual`.
- A moment backed by both modalities can outrank a stronger single-modality
  match — that agreement is the point of fusion (covered by a dedicated test).
- Weights default to 0.5/0.5 (`multimodal_text_weight` / `multimodal_visual_weight`,
  **not tuned** — Phase 10's job) and can be overridden per request; negative
  weights are rejected (422).
- Edge cases handled explicitly: empty modality (no frames indexed yet), a single
  result or identical scores (no spread → each gets 1.0, not 0/0), frames from a
  different video never merge into a chunk, `top_k` truncation after fusion with a
  wider candidate pool.
- Frontend: a **Fused search** tab with a transcript↔visual balance slider,
  evidence-type badges, per-modality score split, matched-frame thumbnails, and
  click-to-seek.
- Verified: 10 mocked unit tests (normalization, merge, tie-break, weights, video
  isolation), 5 integration tests (real ChromaDB in a temp dir, both collections,
  real HTTP endpoints, only the two query encoders faked), and live in the real
  app on the coffee-cake video (query "cinnamon sugar sprinkled on the batter"
  returned a `both` result with the cinnamon-swirl frame; clicking it seeked the
  player to the chunk start).
- **Known limitation (motivates the content-aligned chunking work)**: time
  alignment is only as tight as the chunks. The coffee-cake video is indexed as 2
  chunks of ~200 s each, so almost any frame lands "inside" a chunk and the `both`
  match is loose. Chunk boundaries are word-count-driven, not content-driven.

## Phase 6b — Scene-aware segmentation (SSIM + content-aligned chunking) (complete)

Two changes, motivated by one observation: text chunks were cut every ~150
words and frames were deduplicated with a raw pixel difference — neither knew
what was actually on screen, so a chunk could straddle two slides.

**1. SSIM scene detection** (`frame_service._ssim`, `_is_new_scene`).
Structural Similarity (Wang et al. 2004; 11×11 Gaussian window, σ 1.5) compared
against the last *saved* frame on the existing 64×64 grayscale copy — a frame is
a new scene when SSIM < `frame_ssim_threshold` (0.6). Built on `cv2.GaussianBlur`
(no new dependency) and verified against `skimage.metrics.structural_similarity`
(max difference 6.6e-15 over 40 real frame pairs). `frame_diff_method` selects
`"ssim"` (default) or `"mad"` (the old metric, kept as a benchmark baseline).
Every saved frame after the first marks a scene change, exposed as
`frame_service.get_scene_boundaries()`.

**2. Content-aligned chunking** (`chunking_service`). When frames exist, scene
boundaries become preferred chunk cuts: each boundary is mapped to the transcript
segment whose start is nearest the estimated change time (boundary − half a
sample interval), so timestamps stay exact. `chunk_target_words` (150) remains a
ceiling; `chunk_min_words` (40) is a floor so rapid cuts cannot shred the
transcript. A scene cut carries **no overlap** (overlap exists to protect ideas
split by an *arbitrary* cut). With no frames — or `chunk_align_to_scenes=False` —
behaviour is the original algorithm exactly. `ChunkSet` records `alignment`,
`scene_boundaries_available` and `scene_cuts_applied`. The frontend adds an
"Align text chunks to scenes" pipeline step and skips already-finished steps.

### What the measurements actually showed (including where the plan was wrong)

Reproduce with `backend/scripts/segmentation_benchmark.py` and
`backend/scripts/retrieval_eval.py`.

- **SSIM ≈ MAD at finding boundaries on this data.** On the 254 consecutive
  CS101 sample pairs they disagree on 1. The differences in kept-frame counts
  come from threshold choice, not from SSIM finding different scenes.
- **SSIM's advantage is real but narrow.** A uniform +25 brightness shift scores
  MAD 25 ("new scene") but SSIM 0.93 ("same"); the one real disagreement in the
  data is the lecturer's name-caption overlay (MAD 15.1, SSIM 0.81) — correctly
  *not* a new scene under SSIM.
- **SSIM has a blind spot, found by a failing test:** between two *flat* frames
  (black frame, plain title card) it has no structure to compare — a red→blue
  solid cut scores 0.67, "same scene". Fixed with a narrow fallback: if both
  frames are flat (gray std < 5), compare mean luminance (gap ≥ 20 = new scene).
- **Correction to an earlier claim.** The 79–98% frame keep-rate on the news,
  cooking and vlog videos was first read as pixel-difference noise that SSIM
  would fix. It is not: consecutive samples 5 s apart have median SSIM ≈ 0.35
  there, i.e. the shots genuinely change. The keep-rate reflects the content, and
  SSIM keeps 63–89% on those videos depending on threshold.
- **Aligned chunking makes MORE, smaller chunks, not fewer** (CS101: 20 → 39
  chunks, 155 → 74 words each). The expected "leaner index" did not materialise;
  the index is still tiny. The benefit is elsewhere (below).

**Retrieval evaluation** (CS101, 18 questions whose answer spans were read off the
transcript text — independent of both chunkers; hit = chunk covers ≥ 50% of the
answer; MiniLM cosine ranking as in production):

| Strategy | Chunks | Hit@1 | Hit@3 | MRR | Words sent to the LLM (top-5) |
|---|---|---|---|---|---|
| word-count, ceiling 150 (previous default) | 20 | 83% | 100% | 0.92 | 772 |
| word-count, ceiling 75 (size-matched control) | 42 | 67% | 94% | 0.80 | 408 |
| **scene-aligned** | 39 | **89%** | 100% | **0.94** | **365** |

Reading it honestly: scene-aligned matches the old accuracy (the one-question gap
over word-count@150 is within noise) while giving the LLM **about half the
context** (365 vs 772 words); and against a plain-word-count control of the *same*
chunk size it is clearly better (Hit@1 89% vs 67%, i.e. 16 vs 12 of 18). So the
gain is not merely "smaller chunks". Limits: 18 questions, one video, labels
written by the developer; answer *quality* with the smaller context was not
measured; structural metrics from the segmentation benchmark (alignment rate,
scene purity) are measured against the same boundaries the chunker follows and
only confirm it does what it was built to do.

**Known limitations:** comparison is on grayscale (a colour-only change is
missed); boundaries are accurate to the 5 s sampling interval; frame extraction
decodes sequentially and takes roughly real-time on CPU; on rapid-cut footage
(cooking, vlog) the word floor dominates so aligned chunking degrades to
word-count chunking by design.


## Phase 11 (partial) — Retrieval accuracy: reranking, hybrid search, relevance gate

Question asked: where can accuracy be improved, and are reranking / hybrid search
needed? Method: measure first, adopt only what the data supports. Reproduce with
`backend/scripts/retrieval_ablation.py`, `rerank_analysis.py` and `wer_eval.py`.

**Test set.** 57 labelled questions over 6 videos (`scripts/eval_data/questions.json`):
keyword, detail and *paraphrase* questions (the last deliberately share few words
with the answer). Answer spans were read off the transcripts by the developer.
Scored on the production chunks, per-video (like the UI) and pooled across every
video. A chunk hits when it is from the right video and covers >= 50% of the span.

**What was adopted — cross-encoder reranking.** Embedding retrieval fetches a
shortlist of 10, then `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params) reads
query and chunk *together* and reorders it (`rerank_service`, `settings.retrieval_rerank`).

| (production setting, shortlist 10) | Hit@1 | MRR | vs dense (better/worse questions) |
|---|---|---|---|
| per-video: dense | 68% | 0.790 | — |
| per-video: dense + rerank | **79%** | **0.849** | +9 / −2 |
| pooled: dense | 65% | 0.760 | — |
| pooled: dense + rerank | **77%** | **0.827** | +10 / −2 |

The gain is concentrated where embeddings are weakest — paraphrase questions
(MRR 0.53 → 0.75); keyword questions were already strong (0.91 → 0.96). Cost:
median **405 ms** per query on this CPU (query embedding + scoring 10 chunks, warm;
p95 493 ms). A shortlist of 5 or 10 is about as accurate as 20 at half the cost;
the larger `ms-marco-MiniLM-L-12-v2` was no better than L-6.

**What was tested and NOT adopted — hybrid (BM25 + dense) search.** A dependency-free
BM25 (`lexical_service`) fused with the embedding ranking (Reciprocal Rank Fusion, and
a min-max weighted sum). Per-video MRR 0.790 → 0.806 (RRF, +5/−4 questions) and 0.832
(weighted sum, +6/−5): inside the noise of a 57-question set, and with the reranker
present hybrid added nothing (0.864 vs 0.865 per-video; pooled 0.843 vs 0.830 with
more losses than the reranker alone). Dense retrieval already handles keyword questions
well on short chunks, and the misses that remain (e.g. "yottabyte", transcribed
"Yatsubite") are transcription errors that no retriever can fix. Hybrid was therefore
**not wired into the app**; `lexical_service` and the ablation remain as the evidence.
Worth re-testing if the library grows to thousands of chunks.

**Relevance gate (new).** If even the best reranked chunk scores below
`rag_min_relevance` (−7, raw logit), RAG answers "not enough information" without
calling the LLM. The reranker's top score separates answerable from out-of-scope
questions with AUC 0.987; at −7 all 16 off-topic test questions were refused while
~93% of the 57 answerable ones passed. Side benefit: refusals take **0.2–0.4 s**
instead of 15–32 s for an answered question (the CPU LLM dominates).
*Limits:* the 16 negatives are blatantly off-topic (France, Hamlet, vaccines); a
near-topic unanswerable question would score higher and slip through, and the
answerable questions it wrongly refuses are ones retrieval already fails. Calibrated
on 73 questions — treat the threshold as a starting point, not a result.

**Transcript accuracy (WER).** Only one test video has genuine human captions (a
9.5-minute talk, 1,368 reference words); the cooking clip's "captions" turned out to be
on-screen recipe text, and the others only have auto-captions, so they cannot be scored.
After fixing the scorer (curly apostrophes, digits vs words, HTML entities, which had
inflated early numbers):

| Whisper configuration | WER | CPU time |
|---|---|---|
| base, no options (before the transcript work) | 5.3% | 55 s |
| base, current settings (VAD, no conditioning) | 5.0% | 38 s |
| base + domain prompt | 4.8% | 43 s |
| small | 4.2% | 115 s |

About 40 of the errors in *every* row are one closing outro that is spoken but absent
from the captions, so the true rate is roughly 2%. Reading it honestly: base is already
very accurate on clean speech; the earlier settings changes (VAD etc.) moved WER by ~4
words — within noise — because their purpose (avoiding hallucination over silence/noise)
is not exercised by clean audio; `small` gains ~0.8 points for ~3x CPU time, so `base`
stays the default. Proper-noun errors on noisy or fast speech (e.g. news: "duty rebels")
are real but unmeasured — no reference exists for them.

**Not yet addressed:** answers that straddle a chunk boundary (one test question has no
single covering chunk); some "misses" are ambiguous questions with several valid
evidence spans (the vlog's breakfast); LLM answer quality (qwen2.5:3b) was only
spot-checked; one-developer-written labels and a 98-chunk corpus mean small differences
should be read as noise.


## Phase 6b — UPDATE: pilot results that did not replicate (read this before the section above)

The Phase 6b write-up above reports that scene-aligned chunking "beats a size-matched
control (89% vs 67% Hit@1)". That came from 18 CS101 questions. Re-tested on the larger
57-question set and against independent ground truth, **it does not hold**:

**1. Retrieval over 57 questions (dense only, `scripts/chunking_strategies_eval.py`):**

| Chunking | Chunks | Words/chunk | Hit@1 | MRR | Words sent to LLM |
|---|---|---|---|---|---|
| word-count 150 | 37 | 151 | 74% | 0.852 | 670 |
| word-count 75 (size control) | 78 | 79 | 70% | 0.800 | 381 |
| scene-aligned (SSIM) | 82 | 63 | 61% | 0.750 | 325 |
| topic-aligned (text only) | 42 | 128 | 67% | 0.806 | 576 |
| scene + topic | 83 | 63 | 58% | 0.729 | 323 |

Per video (Hit@1, scene-aligned vs word-count 150 / 75): CS101 (n=33) 73% vs 73% / 67%;
vlog (n=13) **31%** vs 69% / 54%. The 22-point advantage shrank to ~2 questions on CS101,
and on rapid-cut footage scene alignment clearly *hurts*. Cooking (n=6) and news (n=5) are
too small to read.

**2. Boundaries vs YouTube chapter markers** (`scripts/boundary_eval.py`; F1 at +/-15 s).

*Provenance correction.* An earlier version of this section called every chapter list
"creator-published human ground truth". That was **wrong**: parsing each video's
description showed that only some chapters are creator-written timestamps (Indently,
Tarzia CS-310, Paint Explainer). The rest (Core Dumped, Telusko, TED explainer, the
Networks lecture, Soper, Neso, EECS2011) have no timestamps in the description, so they are
most likely **YouTube's auto-generated chapters** -- machine-made, possibly derived from
the transcript, which would favour text-based signals. `chapters.json` now records
`provenance` per video and the evaluation reports the two groups separately.

| Macro F1, +/-15 s | creator-verified (1 video: Indently-30, 19 boundaries) | unverified/auto labels (4 videos, 23 boundaries) |
|---|---|---|
| random, same count as hybrid | 0.283 | 0.237 |
| fixed 60 s windows | 0.245 | 0.349 |
| scene-MAD (visual) | 0.370 | 0.204 |
| scene-SSIM (visual) | 0.296 | 0.166 |
| PySceneDetect (visual) | 0.375 | 0.100 |
| speech pauses | 0.000 | 0.098 |
| topic shift (transcript only) | 0.417 | 0.386 |
| hybrid visual+topic+pause (leave-one-video-out) | 0.426 | 0.349 |

Per video (unverified labels, F1 at +/-15 s): the **slide lecture** (Computer Networks L1)
-- scene-MAD 0.44, scene-SSIM 0.38, topic 0.48, hybrid 0.34, fixed 0.21, random 0.19
(PySceneDetect found no cuts at all there); the animated explainer, Telusko and Core Dumped
-- scene-SSIM 0.12 / 0.00 / 0.16 against random 0.28 / 0.21 / 0.26, with visual detectors
firing 62-84 times against 5-7 chapters.

**What this supports, honestly.** Visual boundaries look **style-dependent**: clearly above
chance on the slide lecture (0.38-0.44 vs 0.19) and the verified code screencast (0.30-0.37
vs 0.28), at or below chance on animated/fast-cut footage where they over-segment. In no
video do they beat the transcript topic signal, and the hybrid never beat its best single
signal. The caveats are large: only ONE video has verified human labels, the slide-lecture
labels are unverified, and the topic signal's lead may be inflated if auto-chapters were
derived from transcripts. At +/-30 s fixed 60 s windows win (0.57-0.61) simply because
chapters are ~60-100 s apart and the tolerance is generous -- the +/-15 s rows are the
meaningful ones. Chapter *purity* of scene-aligned chunks equals that of size-matched
word-count chunks (0.949 vs 0.946; mixed-provenance labels): the "purity" gain reported
earlier was a chunk-size effect.

**Conclusion.** Scene-aligned chunking is **not** better than plain word-count chunking for
retrieval on the 57 questions, and clearly worse on rapid-cut footage. Whether visual
boundaries help on slide lectures specifically is **still open**: the boundary evidence is
encouraging but weakly labelled, and no retrieval questions exist yet for the slide
lecture. `settings.chunk_align_to_scenes` is a candidate to default OFF; the stored indexes
of the four demo videos are still scene-aligned.

**3. A positive result — citation localization** (`scripts/localization_eval.py`; 57
questions; full retrieval stack). Jumping to the start of the cited chunk can leave the
viewer far from the answer; re-scoring the transcript *segments inside* the retrieved chunk
with the cross-encoder and jumping to the best one lands much closer:

| Cited chunks | Jump target | Median error | Within 10 s | Within 20 s |
|---|---|---|---|---|
| plain word-count 150 (mean 79 s long) | chunk start | 25.2 s | 28% | 44% |
| | best segment inside chunk | **7.1 s** | **58%** | **74%** |
| production chunks (mean 37 s long) | chunk start | 8.6 s | 51% | 68% |
| | best segment inside chunk | **5.7 s** | **65%** | **75%** |

Restricted to questions whose top chunk really contains the answer: median error 25.2 s ->
4.7 s (plain) and 6.4 s -> 0.5 s (production). Decoupling retrieval granularity (large
chunks, good recall) from localization granularity (single segments) is the most promising
direction found. Limits: answer spans hand-labelled to a few seconds, one labeller, 57
questions, not yet in the app's citations.

### Bug found while preparing the project report: stale vectors after re-indexing

`vector_store_service.upsert` only overwrites the ids present in the new batch. After
re-extracting frames with SSIM (fewer frames per video than the old pixel-difference
run), the old, higher-numbered frame records stayed in the `video_frames` collection:
294 records were stored for 246 real frames, so visual and fused search could return
frames whose image files no longer existed (broken thumbnails in the UI). The text
collection was unaffected only because re-chunking happened to produce *more* chunks.
Fix: `vector_store_service.delete_video()` and both index functions now delete a
video's existing records before upserting, making re-indexing a true replace. Covered
by two unit tests (delete happens first) and two real-ChromaDB tests (only the target
video is removed; re-indexing with fewer records leaves none stale). All videos were
re-indexed and the counts now match the manifests (198 chunks, 246 frames). The suite
has 167 tests.

## Usability and correctness pass: localized citations, delete, de-duplication

**Citations land on the answer, not the chunk start.** A retrieved chunk spans roughly
40-70 s. The localization study (`backend/scripts/localization_eval.py`) found that
jumping to the chunk start was a median 25.2 s away from the labelled answer span on
150-word chunks, versus 7.1 s when the segments inside the chunk are re-scored by the
cross-encoder and the best one is chosen. That is now in the app: `RetrievalService.search`
sets `jump_time` on the top `localize_top_n` (3) results, `RagService` copies it onto each
`SourceRef`, and the UI seeks to `jump_time ?? start_time` while still showing the whole
passage range. Controlled by `localize_citations` / `localize_top_n`. It is best-effort: if
the transcript is missing or scoring fails, `jump_time` is `None` and the client falls back
to the chunk start. Fused search skips it (its results are time-aligned fusions, not chunks).
Caveat: the 7.1 s figure comes from the hand-labelled study set, not from live user queries.

**Deleting a video.** `DELETE /api/videos/{id}` (`library_service.delete_video`) removes the
stored file, metadata, transcript, chunks, embeddings, audio, the frame directory and both
ChromaDB collections' records for that id. The id is validated against `^[A-Za-z0-9_-]+$`
before any path is built. The library shows a delete button with a confirmation prompt.

**Same link twice.** `POST /api/videos/from-url` now canonicalises the URL
(`youtube:<11-char id>` for watch/youtu.be/shorts/embed forms, otherwise the URL without
fragment and trailing slash). A repeat returns the existing video with HTTP 200 instead of
downloading again (201); the UI says "already in your library".

**UI.** Chat auto-scrolls to the newest answer and shows an elapsed-seconds timer while the
local model generates; search panels show "Searching..." and a first-call model-load hint.

**Verified live** (backend + Vite dev server): a 3 s test upload deleted from the UI left no
files on disk; `youtu.be/...?si=` form of an existing link returned 200 with the library
count unchanged; a RAG answer scrolled into view and its first citation chip seeked the
player to 6:52 for a chip labelled 6:51.

**Tried and rejected: seeking for frame extraction.** Jumping with `CAP_PROP_POS_FRAMES`
instead of decoding sequentially was only 1.2x (Python tutorial) to 1.7x (short news clip)
faster, and on the news clip the sampled frames differed (mean absolute pixel difference
6.25, max 85.6) because seeks land on the nearest keyframe. Sequential sampling stays.

The suite now has 185 tests and the API 27 endpoints.

## Video summary and key points

`summary_service` (`GET`/`POST /api/videos/{id}/summary`, "Summary" tab). It is a small
map-reduce sized for a 3B model on CPU: the transcript is cut into sections of about
`summary_section_words` (350) words at segment boundaries; each section gets one LLM call
for a one-sentence key point; one last call writes a 3-4 sentence overview from those
points. Long videos widen the sections instead of adding calls (`summary_max_sections`
= 10), so the call count is bounded at about 11.

As in RAG, the timestamps are not the model's: each key point's start/end are the real
times of the first and last transcript segment in its section, so the player jump is
exact to the section. The model only writes sentences. Replies are tidied
(`clean_sentence`: stray labels, bullets, quotes), sections under 15 words are skipped, and
the result is cached in `storage/summaries/{id}.json` (removed when the video is deleted)
because generation takes minutes. Limits: key points are per section, not per topic, so a
topic change in the middle of a section is not detected; the summary is only as good as
the transcript and the 3B model. It is not yet evaluated against reference summaries.

Measured on the 9-minute TED talk (1,400 words, 4 sections): 73 s end to end on this CPU,
four key points with section times (0:06-2:21, 2:22-4:34, 4:35-7:04, 7:05-9:23), and a
3-sentence overview. Reading them against the transcript they are on topic, but the last
point is generic ("explained complex ideas using metaphors") rather than a specific claim,
which is the typical weakness of a 3B model. The suite now has 200 tests and the API 29
endpoints (the earlier "185 / 27" figures above predate this feature).
