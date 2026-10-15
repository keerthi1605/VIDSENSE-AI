import { useEffect, useState, useCallback } from "react";
import { api } from "./api";
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
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false));
    refreshVideos();
  }, [refreshVideos]);

  const [notice, setNotice] = useState(null);

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
        <div className="banner banner-error">
          Can't reach the backend at http://127.0.0.1:8000. Start it with{" "}
          <code>uvicorn app.main:app --reload</code> from the <code>backend/</code> folder.
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
          <UploadPanel onVideoAdded={handleVideoAdded} />
          <VideoLibrary
            videos={videos}
            selectedVideoId={selectedVideo?.video_id}
            onSelect={handleSelectVideo}
            onDelete={handleDeleteVideo}
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
