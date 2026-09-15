import { useState } from "react";
import { api } from "../api";

export default function SearchPanel({ video, onSeek, formatTime }) {
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
      const response = await api.search(query.trim(), video.video_id, 5);
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
          placeholder="Semantic search over the transcript (no LLM)"
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
        <ul className="result-list">
          {results.length === 0 && <p className="hint">No results.</p>}
          {results.map((r) => (
            <li key={r.chunk_id} className="result-card" onClick={() => onSeek(r.start_time)}>
              <div className="result-meta">
                <span className="result-time">
                  ▶ {formatTime(r.start_time)}–{formatTime(r.end_time)}
                </span>
                <span className="result-score">score {r.score.toFixed(3)}</span>
              </div>
              <p className="result-text">{r.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
