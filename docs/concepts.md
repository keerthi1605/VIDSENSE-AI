# VidSense AI — Concepts & Viva Preparation

This document is different from [`architecture.md`](architecture.md). That file is a
**build log**: what exists, which file does what, how it was verified. This file is a
**knowledge reference**: the underlying ideas you need to actually understand and defend
in a viva — what each technology is, why it exists, how it works, what the alternatives
were, and why we picked what we picked *for this project, on this hardware*.

Read it phase by phase. Each phase section assumes you've read the ones before it — the
project builds concepts on top of each other, same as the code does.

---

## Table of contents

1. [Phase 1 — Speech Recognition (Whisper)](#phase-1--speech-recognition-whisper)
2. [Phase 2 — Chunking & Text Embeddings](#phase-2--chunking--text-embeddings)
3. [Phase 3 — Vector Databases & Semantic Search](#phase-3--vector-databases--semantic-search)
4. [Phase 4 — RAG & LLMs](#phase-4--rag--llms)
5. [Phase 5 — Computer Vision & CLIP](#phase-5--computer-vision--clip)
6. [Cross-cutting principles](#cross-cutting-principles)
7. [Glossary](#glossary)
8. [Novelty — to be developed](#novelty--to-be-developed)

---

## Phase 1 — Speech Recognition (Whisper)

### What we built
`audio_service.py` extracts mono 16kHz WAV from a video via FFmpeg; `transcription_service.py`
runs Faster-Whisper on that audio, producing a list of `(start_time, end_time, text)`
segments — see [architecture.md](architecture.md#current-state-phase-1-step-3--phase-1-complete).

### The core problem
Traditional video search relies on metadata a human typed in — title, description, tags.
None of that describes *content*. "Where does the professor explain deadlock prevention?"
has no answer unless something has actually listened to the audio and made it searchable.

### What is ASR (Automatic Speech Recognition)?
A model that maps a waveform (a sequence of audio samples) to text. Two families:
- **Hybrid systems** (old-school): separate acoustic model, pronunciation dictionary,
  language model, glued together (e.g. classic Kaldi pipelines).
- **End-to-end neural systems** (modern): one model, trained directly on
  (audio, transcript) pairs, no hand-built dictionary. Whisper is this kind.

### How Whisper actually works
Whisper is an **encoder-decoder Transformer** (the same architecture family as machine
translation models):
- The **encoder** takes a log-Mel spectrogram of the audio (a 2D time-frequency
  representation, not the raw waveform) and produces a sequence of hidden vectors.
- The **decoder** generates text token-by-token, autoregressively, attending back to the
  encoder's output at every step (cross-attention) — same generation mechanism as an LLM,
  just conditioned on audio instead of on text.
- It was trained on **680,000 hours** of multilingual, multitask audio scraped from the
  web with weak supervision (not hand-labeled) — this scale is *why* it generalizes well
  to accents, background noise, and domain-specific vocabulary without fine-tuning.
- Segment timestamps come from Whisper's own internal chunking (it processes audio in
  ~30-second windows) combined with cross-attention alignment — this is why our
  timestamps are segment-level, not word-level, unless you explicitly request
  `word_timestamps=True` (higher compute cost, not enabled here).

### Faster-Whisper: why not the original OpenAI `whisper` package?
Same model weights, different **inference engine**:
- OpenAI's `whisper` runs on PyTorch.
- Faster-Whisper runs on **CTranslate2**, a C++ inference engine purpose-built for
  Transformer models, with better memory layout and kernel fusion.
- Result: **2-4x faster, less RAM**, on identical hardware — the difference between
  "usable" and "painful" on a CPU-only machine.
- We also use **int8 quantization**: model weights are stored as 8-bit integers instead
  of 32-bit floats. This roughly quarters memory usage and speeds up CPU matrix
  multiplication, at a small, generally imperceptible accuracy cost. See
  [Quantization](#quantization) in the glossary — the same idea reappears in Phase 4.

### Model size tradeoff
Whisper ships in `tiny` / `base` / `small` / `medium` / `large` sizes (39M to 1.5B
parameters). We use **`base`**: on an 8GB-RAM CPU-only machine, `medium`/`large` would be
too slow for interactive use, and `tiny` sacrifices real accuracy. This is a config value
(`WHISPER_MODEL_SIZE` in `.env`), not a hard-coded choice — swappable per deployment.

### Alternatives (and why not)
| Option | Why not chosen here |
|---|---|
| `wav2vec2` (Meta) | Strong for single-language fine-tuned use cases, but Whisper's multilingual/robustness-by-scale approach needs no fine-tuning for our general "any lecture" use case |
| Cloud ASR APIs (Google STT, AWS Transcribe, AssemblyAI) | Would work, often more accurate — but costs money per minute, needs network, and violates the project's "local/free by default" hardware-aware design. Also a real dependency risk for an offline demo. |
| Classic hybrid (Kaldi) | Requires building a pronunciation lexicon and language model per domain — far more setup for worse generalization than a large pretrained end-to-end model |

### Ingesting a video from a URL (Phase 1, Step 4)
On top of direct upload, `POST /api/videos/from-url` downloads a video via `yt-dlp`
(YouTube and most other video sites) and stores it **identically** to a direct
upload — same `VideoMetadata` shape, same file layout — so every later phase
(transcription, chunking, embeddings, frames) needs zero awareness of how the video
arrived. It probes duration/liveness *before* downloading (reject early, don't waste
bandwidth), and caps resolution at 720p for the same CPU/storage reasons as everything
else in this project.

A real debugging example worth knowing cold: `yt-dlp`'s `ffmpeg_location` option
does **not** resolve a bare command name via PATH the way a shell does — passing it
`"ffmpeg"` failed with *"ffmpeg is not installed"* even though `ffmpeg` was on PATH
and working everywhere else in the project (audio extraction, frame remuxing). The
fix was resolving it explicitly first, with Python's `shutil.which("ffmpeg")`, and
only passing a result if one was found. **The general lesson, not just this one bug**:
different tools/libraries make different, sometimes-undocumented assumptions about
what a "command name" setting means — never assume PATH resolution is universal
just because it worked for one tool (FFmpeg itself, invoked directly) but not another
(yt-dlp, invoking FFmpeg *for* you via a config option).

### Known limitations (say these out loud in a viva, don't wait to be asked)
- Whisper can **hallucinate** text during silence or non-speech audio (music, noise) —
  a known, documented failure mode, not specific to our implementation. **Observed for
  real** while testing Step 4: a downloaded Creative-Commons animated short with no
  dialogue produced repeated hallucinated "I'm sorry" text at each ~30-second internal
  window boundary — Whisper's chunking interval — a live instance of exactly this
  limitation, not a hypothetical one.
- No **speaker diarization** — we don't know *who* said something, only *what* and *when*.
- Segment-level, not word-level, timestamps — "jump to video" is accurate to a few
  seconds, not to the exact word.
- Accuracy depends on `base` model size — a heavy-accent or highly technical-jargon
  lecture will transcribe worse than a clear, common-vocabulary one.

### Viva questions to be ready for
- *"Why Whisper and not a simpler keyword-spotting model?"* → we need full transcription,
  not detecting a fixed vocabulary; Whisper generalizes to any topic.
- *"What's actually different between Faster-Whisper and Whisper?"* → same weights,
  different **inference engine** (CTranslate2 vs PyTorch) plus quantization — not a
  different model.
- *"Why segment-level and not word-level timestamps?"* → word-level requires extra
  alignment computation Whisper supports but we didn't enable, given segment granularity
  is sufficient for "jump to this part of the video."

---

## Phase 2 — Chunking & Text Embeddings

### What we built
`chunking_service.py` merges consecutive Whisper segments into ~150-word chunks with
1-segment overlap; `embedding_service.py` turns each chunk's text into a 384-dimensional
vector via Sentence-Transformers. See
[architecture.md](architecture.md#phase-2-step-1--transcript-chunking).

### Why chunk at all?
Two independent reasons:
1. **Embedding granularity** — embedding an entire 8,000-word transcript as one vector
   would average away everything specific; a query about one topic would get a vague,
   diluted similarity score against a vector that's "about" the whole lecture.
2. **LLM context limits** (relevant from Phase 4 onward) — an LLM prompt can only hold so
   much text; you must retrieve a *small, relevant* slice, not the whole transcript.

### Chunking strategies (know the landscape, not just what we did)
- **Fixed-size character/token chunking** — split every N characters/tokens. Simple,
  but blind to sentence boundaries; can cut a sentence in half and requires
  interpolating a timestamp for the cut point (a guess, not a measurement).
- **Sentence-based chunking** — split on sentence boundaries, group N sentences per
  chunk. Better, but plain text has no natural "sentence" unit as clean as ours.
- **Semantic chunking** — use embedding similarity between adjacent sentences to decide
  where topics shift, splitting there. More sophisticated, more compute, harder to make
  deterministic.
- **What we did: segment-merge chunking** — merge Whisper's own segments (which are
  already natural speech units with *exact* timestamps) until a word-count target is
  hit. This means **every chunk boundary is a real, measured timestamp** — never
  interpolated. This is the single most important design decision in Phase 2 to be able
  to explain and defend.
- **Overlap**: the last segment of chunk N is repeated at the start of chunk N+1, so an
  idea spanning a boundary appears in both chunks' embeddings and can't fall through a
  retrieval gap.

### What is an embedding, actually?
A **fixed-length vector of real numbers** produced by a neural network, such that
semantically similar inputs produce vectors that are *close together* in that
high-dimensional space, and dissimilar inputs produce vectors that are *far apart*. No
individual number in the vector is human-interpretable — only relative position/distance
carries meaning.

### How is a *useful* text embedding trained?
Not just any neural network output is a good embedding. Sentence-Transformers models
(like our `all-MiniLM-L6-v2`) are trained with a **contrastive objective**: shown pairs of
sentences labeled similar/dissimilar (e.g. via Natural Language Inference datasets, or
mined paraphrase pairs), the model is optimized — typically via a **Siamese network**
(the same encoder run twice, once per sentence, weights shared) — so that cosine
similarity between the two outputs matches the label. This is what makes a *general
sentence encoder* different from, say, raw BERT: raw BERT was trained for masked-word
prediction, not for producing comparable sentence-level vectors — its untuned output is
mediocre for similarity search.

`MiniLM` specifically is a **distilled** model: a smaller "student" network trained to
mimic a larger "teacher" model's behavior, recovering most of the teacher's quality at a
fraction of its size/speed cost — exactly the tradeoff we want on a CPU-only machine.

### Vector similarity: cosine, dot product, Euclidean
- **Euclidean distance**: straight-line distance between two points. Sensitive to vector
  magnitude — a "longer" vector looks farther away even if it points the same direction.
- **Dot product**: sum of element-wise products. Combines both direction *and*
  magnitude — a longer vector can score higher even if less semantically aligned.
- **Cosine similarity**: the *angle* between two vectors, ignoring their length —
  `cos(θ) = (A · B) / (|A| |B|)`. This is what Sentence-Transformer models are actually
  trained/evaluated against, so it's the metric that matches the model's own learned
  notion of "similar."
- **Why we normalize vectors to unit length at embedding time**: once every vector has
  length 1, `|A| |B| = 1`, so cosine similarity reduces to a plain dot product — cheaper
  to compute, and exactly what vector databases (Phase 3) optimize their indexes around.

### Dimensionality
The length of the vector (384 for MiniLM, 512 for CLIP later). More dimensions can
encode more nuance but cost more memory/compute and slow down search at scale. It's a
fixed property of the chosen model — not something we tune independently.

### Alternatives (and why not)
| Option | Why not chosen here |
|---|---|
| TF-IDF / BM25 (classic keyword search) | Purely lexical — matches literal words/stems, not meaning. "Repeatedly halving a sorted array" would NOT match a query about "binary search" without shared words. (A **hybrid** of BM25 + embeddings is a legitimate future improvement — see Phase 11's roadmap.) |
| `all-mpnet-base-v2` (bigger Sentence-Transformer) | Higher quality, ~3x slower — a real option once retrieval quality can be *measured* (Phase 10), not before |
| OpenAI/cloud embedding APIs | Would work, but costs money per call, needs network, breaks the "local-first" design |

### Known limitations
- Fixed max input length — very long chunks get truncated by the model internally (150
  words stays comfortably under MiniLM's limit, by design).
- No fine-tuning on lecture-specific vocabulary — a general-purpose model, not adapted
  to this domain.
- Chunk sizes vary (not perfectly uniform words-per-chunk) — the accepted price for
  exact, non-interpolated timestamps.

### Viva questions
- *"Why not just embed each sentence individually?"* → too fine-grained: a single
  sentence often lacks enough context to be a meaningful retrieval unit, and you'd
  multiply the number of vectors to store/search.
- *"What would happen if you didn't normalize vectors?"* → cosine similarity would still
  be computable, just via a slower formula (dividing by magnitudes every time) — the
  *quality* of retrieval wouldn't change, only the compute cost.
- *"Why is chunk_id designed as `{video_id}_{index}` before the vector DB even exists?"*
  → it's engineered to double as the vector DB's primary key later — no separate ID
  scheme needed in Phase 3.

---

## Phase 3 — Vector Databases & Semantic Search

### What we built
`vector_store_service.py` wraps ChromaDB (the only file that imports it);
`retrieval_service.py` provides `index_video()` and `search()`. See
[architecture.md](architecture.md#phase-3--vector-database--semantic-search-complete).

### Why not just loop over vectors in Python (what we did manually in Phase 2's
verification script)?
Because it's `O(n)` per query — compare the query against *every* stored vector, every
time. Fine for a handful of chunks in a test script; hopeless once you have thousands of
chunks across many videos. A vector database exists specifically to avoid that.

### What is a vector database?
A storage system purpose-built to answer: *"given this query vector, which K stored
vectors are closest?"* — fast, at scale, without a linear scan every time. It does this
via an **index structure** built once (or incrementally) ahead of time.

### Approximate Nearest Neighbor (ANN) search — the core idea
Exact nearest-neighbor search (checking every vector) doesn't scale. ANN algorithms trade
a small amount of guaranteed-exactness for massive speed gains. **HNSW** (Hierarchical
Navigable Small World graphs) — what ChromaDB uses internally — builds a multi-layer
graph where each vector is a node connected to its approximate neighbors; the top layer
has few, long-range links for fast coarse navigation, lower layers have many short-range
links for fine-grained accuracy. A query "walks" down through the layers, narrowing in on
the closest region — closer to `O(log n)` than `O(n)`.

### ChromaDB specifics
- Runs as an **embedded, persistent, local** database (a folder on disk,
  `chroma_db/`) — no separate server process to manage, matching the project's
  local-first design.
- Organizes vectors into **collections** (we use one, `video_chunks`); each entry has an
  ID, a vector, the original text ("document"), and arbitrary **metadata** (`video_id`,
  `start_time`, `end_time`) usable for filtering.
- We explicitly configure the collection for **cosine** distance
  (`hnsw:space: cosine`) — Chroma's *default* is squared-L2, which would silently rank
  results differently than the cosine similarity we've validated since Phase 2. This is
  a real, easy-to-miss bug class: using a vector database with its default distance
  metric without checking it matches your embedding model's trained metric.
- We use **upsert**, not `add` — re-indexing the same video overwrites its old vectors
  instead of duplicating them, making indexing idempotent (safe to re-run).

### Alternatives (and why not)
| Option | Why not chosen here |
|---|---|
| FAISS (Meta) | A library, not a database — no built-in persistence/metadata filtering out of the box; you build that yourself. More control, more work. |
| Pinecone / Weaviate / Milvus | Cloud-hosted or server-based — real production options at scale, but add network dependency/cost/complexity not justified for a local semester project |
| pgvector (Postgres extension) | Makes sense once Postgres is already in the stack (Phase 9) — a legitimate future consolidation, not needed yet |

### Known limitations
- Single-node, file-based — no horizontal scaling; fine for one machine's worth of
  lecture videos, not built for web-scale.
- No built-in hybrid (keyword + vector) search — a pure semantic search can occasionally
  miss an exact-term match a keyword search would catch instantly (e.g. a specific
  function name or acronym).
- Approximate search means, in principle, the true #1 nearest neighbor could occasionally
  rank #2 — an accepted tradeoff for speed at scale (not something we've observed to
  matter at our current small data sizes).

### Viva questions
- *"What actually happens differently if you use L2 distance instead of cosine here?"*
  → results would be re-ranked based on vector magnitude differences (e.g. penalizing
  longer chunks) rather than pure directional similarity — could silently degrade
  results without an obvious error.
- *"Why HNSW and not a simpler structure like a KD-tree?"* → KD-trees degrade badly in
  high dimensions (the "curse of dimensionality") — HNSW is specifically designed to
  stay effective at hundreds of dimensions, which is exactly our regime (384/512-dim).
- *"What does `upsert` buy you over `add`?"* → idempotent re-indexing — running the
  pipeline twice on the same video doesn't create duplicate search results.

---

## Phase 4 — RAG & LLMs

### What we built
`llm_service.py` (a swappable `LLMProvider` abstraction, `OllamaProvider` today) and
`rag_service.py` (prompt construction + orchestration). See
[architecture.md](architecture.md#phase-4--rag-question-answering-complete).

### What is RAG, and why not just fine-tune an LLM on the lecture transcripts?
**RAG (Retrieval-Augmented Generation)**: retrieve relevant text first, then hand it to
an LLM as *context* in the prompt, and ask the LLM to answer *using only that context*.
Fine-tuning (retraining the LLM's weights on your data) is the alternative — and it's
generally the wrong tool here:
- Fine-tuning is expensive and slow to update — add one new lecture, retrain again.
- Fine-tuned models still hallucinate; RAG lets you show your work (the retrieved
  chunks), and — crucially — lets you **cite exact sources**, which fine-tuning cannot
  give you at all (the model wouldn't know which of its training examples "caused" a
  given output).
- RAG data can change instantly (upload a new video, embed and index it, it's queryable)
  — no retraining step at all.

### The RAG pipeline, precisely
```
question → embed (MiniLM) → vector search (Chroma) → top-K chunks
         → build a prompt containing those chunks as context
         → LLM generates an answer using ONLY that context
         → attach source chunks (video_id, timestamps) to the response
```

### What is an LLM, at the mechanism level?
A **decoder-only Transformer**, trained to predict the next token given all previous
tokens ("autoregressive" generation), one token at a time, repeatedly appending its own
output and feeding it back in. "Tokens" are sub-word units (not always whole words) — a
tokenizer splits text into these before the model ever sees it. Everything an LLM
"knows" is encoded implicitly in its trained weights, from massive-scale next-token
prediction on internet-scale text — it has no separate "memory" or "database" it
consults; RAG is literally how you give it access to information it wasn't trained on
(or that's more current/specific than its training data).

### Running an LLM locally: Ollama & quantization
- **Ollama** is a local LLM runtime: it manages downloading model weights, running an
  inference server (built on `llama.cpp`'s engine), and exposing a simple HTTP API
  (`/api/generate`) — our `llm_service.py` just does a `requests.post` to it.
- We run **`qwen2.5:3b`** — a 3-billion-parameter instruction-tuned model, chosen after
  checking actual hardware constraints (8GB total RAM, no usable GPU). A bigger model
  (7B+) would answer better but risks starving the machine's RAM alongside everything
  else running.
- **Quantization** (same core idea as Whisper's int8, Phase 1): model weights are stored
  in a lower-precision format than the full-precision training format, trading a small
  amount of output quality for large reductions in memory and (on CPU) real speed gains.
  This is *why* a 3B-parameter model is runnable on 8GB RAM at all.

### Why citations come from retrieval, never from the LLM's own output
This is the single most important engineering decision in Phase 4. A small local LLM
asked to *also* produce structured citations (e.g. "cite chunk_0003") is unreliable —
small models are notoriously bad at strict structured output, and there's no guarantee
the cited ID is even real. Instead: the LLM's **only** job is the natural-language
answer. The timestamps/video IDs shown to the user are attached by our own code
afterward, built directly from the `SearchResult`s that were fed into the prompt — data
we already trust completely. This guarantees every citation is exact, never invented.

### Prompt engineering, briefly
The prompt explicitly instructs the model to (1) use only the provided excerpts, (2)
say so explicitly if the excerpts are insufficient, and (3) be concise rather than
repeating excerpts verbatim. This is the actual mechanism for reducing hallucination —
not a guarantee (a model can still ignore instructions), but a strong steering signal
that measurably changes model behavior in practice.

### Alternatives (and why not)
| Option | Why not chosen here |
|---|---|
| Cloud LLM API (Anthropic/OpenAI/Gemini) | Higher quality, no local resource pressure — but needs an API key, costs money per call, needs network. Explicitly designed to be swappable in later (`llm_service.py`'s whole purpose) — a real option, not rejected forever. |
| Bigger local model (7B/13B+) | Better answers, but 8GB total RAM makes this risky alongside the rest of the stack (torch, sentence-transformers, opencv, chromadb all resident) |
| Fine-tuning instead of RAG | See above — wrong tool for a frequently-changing, citation-requiring use case |

### Known limitations
- No conversation memory — every question is answered independently; no multi-turn
  context.
- `has_sufficient_context` is a **deterministic retrieval check** (did search return any
  chunks at all), not a parse of whether the LLM's own answer text indicates it could
  actually answer — a documented simplification, not a hidden bug.
- Answer quality is capped by a 3B model's real reasoning ability — it can synthesize
  correctly from clear context but isn't going to reason through something genuinely
  subtle.
- No answer-quality evaluation yet (planned for Phase 10): correctness, faithfulness,
  and citation accuracy are currently checked manually, not measured with a metric.

### Viva questions
- *"Why can't the LLM just tell you which chunk it used?"* → small models are unreliable
  at structured/precise output; trusting retrieval data (which we already validated) is
  strictly safer than trusting a generated citation.
- *"What exactly does quantization trade away?"* → numerical precision in the weights —
  in practice a small, usually-imperceptible quality loss, in exchange for a large
  memory/speed win, which is what makes local CPU inference of a multi-billion-parameter
  model feasible at all.
- *"Why RAG instead of just pasting the whole transcript into the prompt?"* → context
  window limits, cost/latency (larger prompts are slower), and dilution — irrelevant
  content in the prompt can measurably degrade an LLM's focus on the relevant part.

---

## Phase 5 — Computer Vision & CLIP

### What we built
`frame_service.py` (OpenCV sampling + duplicate-skip), `vision_service.py` (CLIP
embeddings), and `retrieval_service.search_frames()`/`index_video_frames()` (indexing
frame vectors into a separate ChromaDB collection, `video_frames`, alongside the
text-chunk one). See
[architecture.md](architecture.md#phase-5-step-3--indexing-frames-into-the-vector-store-phase-5-complete).

### Why audio-only search isn't enough
Whisper only knows what was *said*. Slides, code, diagrams, whiteboard content — often
where the actual information density is highest in a lecture — are invisible to it.

### Frame sampling strategies (again, know the landscape)
- **Extract every frame**: correct but wildly wasteful — a 10-minute lecture at 30fps is
  18,000 frames, almost all near-duplicates of a static slide.
- **Uniform/fixed-interval sampling**: one frame every N seconds. Simple, bounded, but
  can still capture many duplicates of a long-static slide, or miss a very brief slide
  shown for less than the interval.
- **Scene-change/shot-boundary detection**: analyze frame-to-frame difference
  continuously, save only on a detected change. More precise, more compute.
- **What we did: fixed-interval + near-duplicate skip (a hybrid)**: sample candidates
  at a fixed cadence (bounding the work), then compare each candidate to the *last
  saved* frame via mean absolute pixel difference on a small downsampled grayscale
  copy — skip if below a threshold. Cheap, deterministic, and empirically verified (in
  our tests) to correctly tell a static scene from a real change.

### Why OpenCV, not FFmpeg, for this step
FFmpeg is excellent at *decoding/transcoding* (used for audio extraction in Phase 1) but
extracting frames via FFmpeg means writing candidate frames to disk first, then reading
them back to compare pixels. OpenCV's `VideoCapture` gives frame-by-frame programmatic
access with real pixel arrays *and* exact timestamps, in a single pass — the right tool
when you need to make a decision (keep/discard) based on the actual image content.

### What is CLIP, and why is it the only real option here?
**CLIP (Contrastive Language-Image Pre-training)**, from OpenAI: two encoders — one for
images, one for text — trained *jointly* via a contrastive loss on hundreds of millions
of (image, caption) pairs scraped from the web. The loss pulls a matching image/caption
pair's embeddings together and pushes non-matching pairs apart. The result: images and
text land in the **same vector space**, where cosine similarity between an image vector
and a text vector reflects how well that text describes that image.

This is not "a vision model we picked" — it's **the only kind of model** that makes
"search images using a text query" possible at all. An ordinary vision model (a
classifier, a plain ResNet) only knows how to compare images to other images; it has no
notion of text whatsoever. Without a joint embedding space, comparing a text query to an
image would be structurally impossible, not just lower quality.

### Two separate embedding spaces — a critical distinction
MiniLM's text-chunk space (384-dim, from Phase 2) and CLIP's joint space (512-dim, from
Phase 5) are **completely different, non-comparable vector spaces**. You cannot compare
a MiniLM vector to a CLIP vector — they were trained independently, with different
objectives, at different dimensionalities. Visual search means embedding the *query* a
**second time**, through CLIP's text encoder specifically, and comparing that against
CLIP's image embeddings — entirely separately from MiniLM-based chunk search. Phase 6
will run both searches independently and fuse the two ranked lists with a weighted score
— it cannot merge the raw vectors themselves.

### Indexing frames: a second, independent vector-database collection
Everything said about ChromaDB and HNSW in [Phase 3](#phase-3--vector-databases--semantic-search)
applies again here, unchanged — the only new idea is that frame vectors get their **own**
collection (`video_frames`), separate from the text-chunk collection (`video_chunks`).
This isn't a stylistic choice: a Chroma collection is built around one fixed vector
dimensionality, and 384-dim MiniLM vectors and 512-dim CLIP vectors could not physically
share one even if we wanted them to — on top of the fact that, per the previous section,
they're non-comparable spaces anyway. This is also why `vector_store_service.py` was
refactored to take an explicit `collection_name` on every call rather than assuming a
single global collection: the same low-level "talk to Chroma" code now serves two
independent, differently-shaped indexes.

### Model choice: `clip-ViT-B-32`
The base/32 ViT (Vision Transformer) CLIP checkpoint — the standard baseline, ~600MB,
moderate compute cost. A larger variant (`ViT-L/14`) is more accurate but meaningfully
heavier; not justified without a way to measure whether it actually improves retrieval
(Phase 10's job). Accessed via `sentence-transformers`, which we already depend on for
MiniLM — same `.encode()` API for both images and text, no new library.

### Alternatives (and why not)
| Option | Why not chosen here |
|---|---|
| BLIP / BLIP-2 | Stronger at generating captions and some VQA tasks, but heavier and more complex to run locally; CLIP's simpler joint-embedding property is exactly what pure retrieval needs |
| OCR (Tesseract) on slide text | A genuinely complementary idea — extracting literal text from a slide image — but solves a different problem (exact text extraction) than semantic image search; a good candidate for a future enhancement, not a CLIP replacement |
| Object detection (YOLO, etc.) | Answers "what objects are in this frame," not "does this frame match this natural-language description" — wrong tool for our retrieval goal |

### Known limitations
- CLIP is weaker at reading **dense text within an image** (e.g. small code on a slide)
  than a dedicated OCR model would be — it "sees" an image holistically, not
  character-by-character.
- Our duplicate-detection threshold (10.0, on a 0-255 pixel-difference scale) is a
  starting heuristic, not tuned against real lecture footage or a formal study.
- Validated so far on solid-color synthetic test images (a clean proof the mechanism
  works) — not yet validated on real, visually complex lecture slides.

### Viva questions
- *"Could you use CLIP for the whole project instead of Whisper?"* → no — CLIP has no
  audio understanding at all; it's purely image/text. The two models solve genuinely
  different problems and are both necessary.
- *"Why can't you combine a MiniLM chunk vector and a CLIP frame vector into one
  search?"* → they're different vector spaces with different dimensionalities (384 vs
  512) trained completely independently — combining them requires running two separate
  searches and fusing the *results* (Phase 6), not the vectors.
- *"What would break if you skipped the near-duplicate check and just used fixed
  interval sampling?"* → nothing would *break*, but a long static slide would produce
  many redundant near-identical embeddings, wasting storage/compute without adding
  retrievable information.
- *"Why two ChromaDB collections instead of one?"* → a collection has one fixed vector
  dimensionality; 384-dim text and 512-dim frame vectors couldn't share one even ignoring
  that they're non-comparable spaces. This is also why the vector-store code was
  refactored to take an explicit collection name on every call instead of assuming a
  single global collection.

---

## Cross-cutting principles

These apply across every phase — a viva panel is very likely to probe whether you
understand these as *deliberate architectural choices*, not accidents.

### 1. Timestamp preservation
Every stage — transcription → chunking → embedding → retrieval → RAG — carries
`start_time`/`end_time` forward **without ever interpolating or approximating** them. The
chunking design (merging whole segments, never cutting mid-segment) exists specifically
so this is possible. This is what makes "jump to the exact moment in the video" a
promise the whole system can actually keep, not just a UI feature bolted on top.

### 2. Model abstraction / swappability
No component hard-codes a specific model or provider. `config.py` centralizes every
model name/setting; `llm_service.py`'s `LLMProvider` ABC is the clearest example — an
Anthropic or OpenAI provider later is a new class and one `if` branch, not a rewrite of
anything that calls it. This is why the project can honestly claim "not hard-coded around
one provider" instead of just asserting it.

### 3. Hardware-aware, local-first design
Every model choice (Whisper `base`, MiniLM, `qwen2.5:3b`, CLIP `ViT-B-32`) was made
*after* checking this machine's actual constraints (8GB RAM, no usable GPU, an old
Vulkan-incompatible discrete GPU) — not by defaulting to "biggest available." This is a
real engineering skill (right-sizing to constraints) worth explicitly narrating in a
viva, including the genuine bugs this surfaced (the Ollama/Vulkan crash, the HuggingFace
`xet` transfer backend instability) and how they were diagnosed from actual logs, not
guessed at.

### 4. Storage separation by kind
- **Raw media** (video/audio/frame files) → filesystem.
- **Structured metadata** (transcripts, chunks, frame manifests) → JSON.
- **Dense vectors** → binary `.npy` files, *never* JSON (size/parse-speed), *never*
  returned raw over the API (not useful to a client, and large).
- **Indexed, searchable vectors** → the vector database (ChromaDB), a distinct concern
  from "where is this vector stored at rest" (the `.npy` file).

This is a directly explainable answer to "why do you have both `.npy` files and a vector
database — isn't that redundant?" → they're not redundant: the `.npy` file is the source
of truth / re-indexable backup, the vector database is the queryable index built from it.

### 5. Explainability
The system can always say *"I answered this based on: 01:24:17–01:25:03"* — every piece
of evidence is traceable to an exact location in a specific video. This is what makes the
RAG answers trustworthy rather than a black box, and it only works because of principle
#1 (timestamps survive every stage) and the Phase 4 decision to source citations from
retrieval, never generation.

---

## Glossary

| Term | Meaning |
|---|---|
| **ASR** | Automatic Speech Recognition — audio → text |
| **Transformer** | The neural network architecture behind Whisper, MiniLM, CLIP, and the LLM — built on the "attention" mechanism, which lets the model weigh the relevance of every other input position when processing each position |
| **Encoder / Decoder** | Encoder: reads input, produces a representation. Decoder: generates output (often token-by-token), sometimes conditioned on an encoder's output (Whisper) or purely on its own past output (the LLM) |
| **Embedding** | A fixed-length vector representing the meaning of an input (text or image), such that similar inputs → nearby vectors |
| **Contrastive learning** | A training method that pulls "similar pair" embeddings together and pushes "dissimilar pair" embeddings apart — how MiniLM and CLIP are trained |
| **Cosine similarity** | Similarity between two vectors based on the angle between them, ignoring magnitude |
| **Quantization** | Storing model weights at lower numeric precision (e.g. int8 instead of float32) to save memory/compute, at a small quality cost |
| **Distillation** | Training a smaller "student" model to mimic a larger "teacher" model's outputs, recovering most quality at a fraction of the cost (how MiniLM was made) |
| **ANN (Approximate Nearest Neighbor)** | A search algorithm that finds *very likely* nearest vectors, much faster than checking every one, at a tiny accuracy tradeoff |
| **HNSW** | Hierarchical Navigable Small World graphs — the specific ANN algorithm ChromaDB uses |
| **RAG** | Retrieval-Augmented Generation — retrieve relevant text, then have an LLM generate an answer grounded in it |
| **Hallucination** | An LLM (or Whisper) confidently producing incorrect/invented output |
| **Token** | A sub-word unit of text; the actual unit LLMs and Whisper's decoder operate on, not always a whole word |
| **Prompt engineering** | Deliberately structuring the instructions/context given to an LLM to steer its behavior |
| **Joint embedding space** | A vector space shared by two different modalities (text and images, for CLIP), enabling direct cross-modal comparison |

---

## Novelty — to be developed

You mentioned wanting to add a novel contribution on top of the base pipeline before this
is finished — noted, deliberately **not designed yet**. When you're ready to tackle it,
some directions worth considering (not commitments — just a starting menu to react to):

- **Hybrid search**: fuse BM25 (keyword) with the existing semantic search, so exact
  terms (function names, acronyms) aren't lost to pure semantic matching.
- **Reranking**: a second-stage cross-encoder re-scores the top-K semantic results
  before they reach the LLM — often a meaningful quality jump for not much extra compute.
- **Cross-modal fusion strategy comparison** (ties into Phase 6): don't just implement
  one fusion formula — measure a couple of weightings/strategies against a small labeled
  test set and report which works better. Turns a design choice into a result.
- **OCR-augmented visual search**: extract literal slide text (Tesseract) alongside
  CLIP embeddings, so exact on-slide text becomes searchable too — a real gap CLIP alone
  has (see Phase 5's limitations).
- **Speaker diarization**: identify *who* is speaking (useful for Q&A sessions,
  multi-speaker recordings) — a real, well-scoped gap noted in Phase 1.
- **A genuine evaluation dataset + metrics** (Phase 10, but could be elevated into "the"
  novelty): most course projects skip rigorous evaluation entirely — doing it well
  (Precision@K, Recall@K, WER, faithfulness scoring) is itself a differentiator many
  peers won't have.

We'll pick one (or combine two) and design it properly — with the same explain → design →
implement → test rhythm as every other phase — when you're ready to revisit this.
