import { useState } from "react";
import { api } from "../api";

const EVIDENCE_LABEL = { text: "Transcript", visual: "Visual", both: "Transcript + Visual" };

export default function MultimodalSearchPanel({ video, onSeek, formatTime }) {
  const [query, setQuery] = useState("");
  // 0 = text only, 1 = visual only; text_weight = 1 - visualShare.
  const [visualShare, setVisualShare] = useState(0.5);
  const [results, setResults] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const response = await api.searchMultimodal(
        query.trim(),
        video.video_id,
        6,
        Number((1 - visualShare).toFixed(2)),
        Number(visualShare.toFixed(2))
      );
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
          placeholder="Fused search: what's said AND what's shown"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !query.trim()}>
          Search
        </button>
      </form>

      <div className="weight-row">
        <span>Transcript</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={visualShare}
          onChange={(e) => setVisualShare(Number(e.target.value))}
          disabled={busy}
          aria-label="Balance between transcript and visual evidence"
        />
        <span>Visual</span>
        <span className="weight-value">
          {Math.round((1 - visualShare) * 100)} / {Math.round(visualShare * 100)}
        </span>
      </div>

      {error && <p className="error-text">{error}</p>}
      {results && (
        <ul className="result-list">
          {results.length === 0 && <p className="hint">No results.</p>}
          {results.map((r) => (
            <li
              key={`${r.chunk_id || ""}${r.frame_id || ""}`}
              className="result-card fused-card"
              onClick={() => onSeek(r.start_time)}
            >
              {r.frame_id && (
                <img
                  className="fused-thumb"
                  src={api.frameImageUrl(video.video_id, r.frame_id)}
                  alt="matched frame"
                />
              )}
              <div className="fused-body">
                <div className="result-meta">
                  <span className="result-time">
                    ▶ {formatTime(r.start_time)}
                    {r.end_time > r.start_time && `–${formatTime(r.end_time)}`}
                  </span>
                  <span className={`evidence-badge evidence-${r.evidence_type}`}>
                    {EVIDENCE_LABEL[r.evidence_type]}
                  </span>
                  <span className="result-score">{r.final_score.toFixed(2)}</span>
                </div>
                {r.text && <p className="result-text">{r.text}</p>}
                <div className="score-split">
                  text {r.text_score.toFixed(2)} · visual {r.visual_score.toFixed(2)}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
