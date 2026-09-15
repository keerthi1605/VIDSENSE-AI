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

  const handleVideoAdded = async (metadata) => {
    await refreshVideos();
    setSelectedVideo(metadata);
  };

  const handleSelectVideo = (metadata) => {
    setSelectedVideo(metadata);
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

      <div className="app-body">
        <aside className="sidebar">
          <UploadPanel onVideoAdded={handleVideoAdded} />
          <VideoLibrary
            videos={videos}
            selectedVideoId={selectedVideo?.video_id}
            onSelect={handleSelectVideo}
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
