import { useRef, useState } from "react";
import { api } from "../api";

export default function UploadPanel({ onVideoAdded }) {
  const [mode, setMode] = useState("file"); // "file" | "url"
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const metadata = await api.uploadVideo(file);
      onVideoAdded(metadata);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  };

  const handleUrlSubmit = async (event) => {
    event.preventDefault();
    if (!url.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const metadata = await api.addVideoFromUrl(url.trim());
      onVideoAdded(metadata);
      setUrl("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel upload-panel">
      <h3>Add a video</h3>
      <div className="tab-row">
        <button className={mode === "file" ? "tab active" : "tab"} onClick={() => setMode("file")}>
          Upload file
        </button>
        <button className={mode === "url" ? "tab active" : "tab"} onClick={() => setMode("url")}>
          From URL
        </button>
      </div>

      {mode === "file" ? (
        <div className="upload-file-area">
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.mov,.mkv,.avi,.webm"
            onChange={handleFileChange}
            disabled={busy}
          />
        </div>
      ) : (
        <form onSubmit={handleUrlSubmit} className="upload-url-form">
          <input
            type="text"
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={busy}
          />
          <button type="submit" disabled={busy || !url.trim()}>
            {busy ? "Fetching…" : "Add"}
          </button>
        </form>
      )}

      {busy && <p className="hint">Working — this can take a moment for a large file or download…</p>}
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}
