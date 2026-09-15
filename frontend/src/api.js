// Thin wrapper around the VidSense AI backend API. One function per
// endpoint, mirroring the backend's own route names -- keeps this file
// a 1:1 map to docs/architecture.md rather than its own abstraction.

const BASE_URL = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON -- keep statusText
    }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

export const api = {
  // --- Videos ---
  listVideos: () => request("/api/videos"),
  getVideo: (videoId) => request(`/api/videos/${videoId}`),
  videoFileUrl: (videoId) => `${BASE_URL}/api/videos/${videoId}/file`,
  uploadVideo: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/videos/upload", { method: "POST", body: form });
  },
  addVideoFromUrl: (url) =>
    request("/api/videos/from-url", { method: "POST", body: JSON.stringify({ url }) }),

  // --- Phase 1: transcription ---
  // initialPrompt/hotwords are optional per-video domain hints (e.g. a
  // technical lecture's course name vs. a cooking video's ingredient
  // list) -- useful when testing across genres. Omitted keys fall back
  // to the backend's settings-level default.
  transcribe: (videoId, { initialPrompt, hotwords } = {}) =>
    request("/api/videos/transcribe", {
      method: "POST",
      body: JSON.stringify({
        video_id: videoId,
        initial_prompt: initialPrompt ?? null,
        hotwords: hotwords ?? null,
      }),
    }),
  getTranscript: (videoId) => request(`/api/videos/${videoId}/transcript`),

  // --- Phase 2: chunking + embeddings ---
  chunk: (videoId) => request(`/api/videos/${videoId}/chunks`, { method: "POST" }),
  getChunks: (videoId) => request(`/api/videos/${videoId}/chunks`),
  embed: (videoId) => request(`/api/videos/${videoId}/embeddings`, { method: "POST" }),

  // --- Phase 3: vector indexing + search ---
  index: (videoId) => request(`/api/videos/${videoId}/index`, { method: "POST" }),
  search: (query, videoId, topK) =>
    request("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, video_id: videoId || undefined, top_k: topK }),
    }),

  // --- Phase 4: RAG chat ---
  chat: (query, videoId) =>
    request("/api/chat", {
      method: "POST",
      body: JSON.stringify({ query, video_id: videoId || undefined }),
    }),

  // --- Phase 5: frames + CLIP + visual search ---
  extractFrames: (videoId) => request(`/api/videos/${videoId}/frames`, { method: "POST" }),
  getFrames: (videoId) => request(`/api/videos/${videoId}/frames`),
  frameImageUrl: (videoId, frameId) => `${BASE_URL}/api/videos/${videoId}/frames/${frameId}/image`,
  embedFrames: (videoId) => request(`/api/videos/${videoId}/frame-embeddings`, { method: "POST" }),
  indexFrames: (videoId) => request(`/api/videos/${videoId}/index-frames`, { method: "POST" }),
  searchVisual: (query, videoId, topK) =>
    request("/api/search/visual", {
      method: "POST",
      body: JSON.stringify({ query, video_id: videoId || undefined, top_k: topK }),
    }),

  health: () => request("/health"),
};
