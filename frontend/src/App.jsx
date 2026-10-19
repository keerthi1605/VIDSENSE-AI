import { useEffect, useState, useCallback } from "react";
import { api, getApiBase, saveApiBase, clearApiBase } from "./api";
import UploadPanel from "./components/UploadPanel";
import VideoLibrary from "./components/VideoLibrary";
import PipelineStatus from "./components/PipelineStatus";
import VideoWorkspace from "./components/VideoWorkspace";
import "./App.css";

export default function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [backendUp, setBackendUp] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [readOnly, setReadOnly] = useState(false);
  const [addressInput, setAddressInput] = useState("");
  const [addressError, setAddressError] = useState(null);

  const refreshVideos = useCallback(async () => {
    try {
      const list = await api.listVideos();
      setVideos(list);
      return list;
    } catch (err) {
      setLoadError(err.message);
      return [];
    }
  }, []);

  useEffect(() => {
    api
      .health()
      .then((h) => {
        setBackendUp(true);
        setReadOnly(Boolean(h && h.read_only));
      })
      .catch(() => setBackendUp(false));
    refreshVideos();
  }, [refreshVideos]);

  const [notice, setNotice] = useState(null);

  // Point the app at a different backend (e.g. a new tunnel address) and start over.
  const handleSaveAddress = (event) => {
    event.preventDefault();
    const problem = saveApiBase(addressInput);
    if (problem) {
      setAddressError(problem);
      return;
    }
    window.location.reload();
  };
  const handleResetAddress = () => {
    clearApiBase();
    window.location.reload();
  };

  const handleVideoAdded = async (metadata) => {
    // Pasting a link that is already in the library returns the existing video
    // (the backend does not download it again) -- say so instead of silently
    // looking like a fresh upload.
    const alreadyThere = videos.some((v) => v.video_id === metadata.video_id);
    setNotice(alreadyThere ? `"${metadata.original_filename}" is already in your library — opened it.` : null);
    await refreshVideos();
    setSelectedVideo(metadata);
  };

  const handleSelectVideo = (metadata) => {
    setNotice(null);
    setSelectedVideo(metadata);
  };

  const handleDeleteVideo = async (video) => {
    const ok = window.confirm(
      `Delete "${video.original_filename}"?\n\nThis removes the video file, its transcript, frames and search index. It cannot be undone.`
    );
    if (!ok) return;
    try {
      await api.deleteVideo(video.video_id);
      if (selectedVideo?.video_id === video.video_id) setSelectedVideo(null);
      setNotice(`Deleted "${video.original_filename}".`);
      await refreshVideos();
    } catch (err) {
      setNotice(`Could not delete: ${err.message}`);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-title">
          <span className="logo-dot" />
          <h1>VidSense AI</h1>
        </div>
        <div className={`backend-badge ${backendUp ? "up" : backendUp === false ? "down" : ""}`}>
          {backendUp === null ? "checking backend…" : backendUp ? "backend online" : "backend offline"}
        </div>
      </header>

      {backendUp === false && (
        <div className="banner banner-error banner-stack">
          <span>
            Can't reach the backend at <code>{getApiBase() || "this site"}</code>. Start it (for local development:{" "}
            <code>uvicorn app.main:app --reload</code> from the <code>backend/</code> folder), or enter its address below.
          </span>
          <form className="address-form" onSubmit={handleSaveAddress}>
            <input
              type="text"
              placeholder="https://your-backend-address"
              value={addressInput}
              onChange={(e) => setAddressInput(e.target.value)}
              aria-label="Backend address"
            />
            <button type="submit">Use this address</button>
            <button type="button" onClick={handleResetAddress}>
              Reset
            </button>
          </form>
          {addressError && <span className="error-text">{addressError}</span>}
        </div>
      )}

      {notice && (
        <div className="banner banner-info" role="status">
          <span>{notice}</span>
          <button className="banner-close" onClick={() => setNotice(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      <div className="app-body">
        <aside className="sidebar">
          {!readOnly && <UploadPanel onVideoAdded={handleVideoAdded} />}
          <VideoLibrary
            videos={videos}
            selectedVideoId={selectedVideo?.video_id}
            onSelect={handleSelectVideo}
            onDelete={handleDeleteVideo}
            readOnly={readOnly}
            error={loadError}
          />
        </aside>

        <main className="main-content">
          {!selectedVideo ? (
            <div className="empty-state">
              <h2>No video selected</h2>
              <p>Upload a video, paste a URL, or pick one from your library to get started.</p>
            </div>
          ) : (
            <>
              <PipelineStatus
                key={selectedVideo.video_id}
                video={selectedVideo}
                readOnly={readOnly}
                onUpdated={(meta) => setSelectedVideo(meta)}
              />
              <VideoWorkspace video={selectedVideo} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
