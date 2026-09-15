import { useEffect, useState } from "react";
import { api } from "../api";

export default function TranscriptPanel({ video, onSeek, formatTime }) {
  const [transcript, setTranscript] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setTranscript(null);
    setError(null);
    api
      .getTranscript(video.video_id)
      .then(setTranscript)
      .catch(() => setError("Not transcribed yet — run the text pipeline above."));
  }, [video.video_id]);

  return (
    <div className="panel transcript-panel">
      <h3>Transcript</h3>
      {error && <p className="hint">{error}</p>}
      {transcript && (
        <ul className="transcript-list">
          {transcript.segments.map((seg) => (
            <li key={seg.segment_id} className="transcript-row" onClick={() => onSeek(seg.start_time)}>
              <span className="transcript-time">{formatTime(seg.start_time)}</span>
              <span className="transcript-text">{seg.text}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
