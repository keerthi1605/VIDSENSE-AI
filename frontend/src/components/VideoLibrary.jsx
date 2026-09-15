function statusBadge(status) {
  const map = {
    uploaded: { label: "uploaded", cls: "badge-neutral" },
    transcribed: { label: "transcribed", cls: "badge-progress" },
    ready: { label: "ready", cls: "badge-ready" },
    failed: { label: "failed", cls: "badge-error" },
  };
  return map[status] || { label: status, cls: "badge-neutral" };
}

export default function VideoLibrary({ videos, selectedVideoId, onSelect, error }) {
  return (
    <div className="panel video-library">
      <h3>Your videos ({videos.length})</h3>
      {error && <p className="error-text">{error}</p>}
      {videos.length === 0 && !error && <p className="hint">Nothing uploaded yet.</p>}
      <ul className="video-list">
        {videos.map((video) => {
          const badge = statusBadge(video.status);
          return (
            <li
              key={video.video_id}
              className={video.video_id === selectedVideoId ? "video-item selected" : "video-item"}
              onClick={() => onSelect(video)}
            >
              <div className="video-item-title" title={video.original_filename}>
                {video.original_filename}
              </div>
              <div className="video-item-meta">
                <span className={`badge ${badge.cls}`}>{badge.label}</span>
                <span className="video-item-size">{(video.size_bytes / (1024 * 1024)).toFixed(1)} MB</span>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
