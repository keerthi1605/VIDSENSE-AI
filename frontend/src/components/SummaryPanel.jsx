import { useEffect, useState } from "react";
import { api } from "../api";

export default function SummaryPanel({ video, onSeek, formatTime }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true); // checking for a cached summary
  const [busy, setBusy] = useState(false); // generating
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setSummary(null);
    setError(null);
    setLoading(true);
    api
      .getSummary(video.video_id)
      .then((s) => !cancelled && setSummary(s))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [video.video_id]);

  useEffect(() => {
    if (!busy) return undefined;
    setElapsed(0);
    const id = setInterval(() => setElapsed((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [busy]);

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      setSummary(await api.createSummary(video.video_id));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const ready = video.status === "transcribed" || video.status === "ready";

  return (
    <div className="summary-panel">
      {loading && <p className="hint">Checking for a saved summary…</p>}

      {!loading && !summary && !busy && (
        <div className="summary-empty">
          <p className="hint">
            Generate an overview and timestamped key points from the transcript. The local model reads the video
            section by section, so this takes a few minutes on CPU. The result is saved.
          </p>
          <button onClick={generate} disabled={!ready}>
            Generate summary
          </button>
          {!ready && <p className="hint">Transcribe this video first (text pipeline).</p>}
        </div>
      )}

      {busy && <p className="hint">Summarising… {elapsed}s (one model call per section, then the overview)</p>}
      {error && <p className="error-text">{error}</p>}

      {summary && (
        <>
          <h4 className="summary-heading">Overview</h4>
          <p className="summary-overview">{summary.overview}</p>

          <h4 className="summary-heading">Key points</h4>
          <ul className="keypoint-list">
            {summary.key_points.map((p) => (
              <li key={p.start_time} className="keypoint" onClick={() => onSeek(p.start_time)}>
                <span className="result-time">
                  ▶ {formatTime(p.start_time)}–{formatTime(p.end_time)}
                </span>
                <span className="keypoint-text">{p.text}</span>
              </li>
            ))}
          </ul>
          <button className="summary-regen" onClick={generate} disabled={busy}>
            {busy ? "Regenerating…" : "Regenerate"}
          </button>
        </>
      )}
    </div>
  );
}
