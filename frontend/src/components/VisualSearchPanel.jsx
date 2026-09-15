import { useState } from "react";
import { api } from "../api";

export default function VisualSearchPanel({ video, onSeek, formatTime }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const response = await api.searchVisual(query.trim(), video.video_id, 6);
      setResults(response.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="search-panel">
      <form onSubmit={handleSubmit} className="chat-input-row">
        <input
          type="text"
          placeholder="Describe what to find visually, e.g. 'a diagram'"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !query.trim()}>
          Search
        </button>
      </form>
      {error && <p className="error-text">{error}</p>}
      {results && (
        <div className="frame-grid">
          {results.length === 0 && <p className="hint">No results.</p>}
          {results.map((r) => (
            <div key={r.frame_id} className="frame-card" onClick={() => onSeek(r.timestamp)}>
              <img src={api.frameImageUrl(video.video_id, r.frame_id)} alt={`frame at ${r.timestamp}`} />
              <div className="frame-card-meta">
                <span>▶ {formatTime(r.timestamp)}</span>
                <span>{r.score.toFixed(3)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
