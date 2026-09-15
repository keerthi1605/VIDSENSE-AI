import { useRef, useState } from "react";
import { api } from "../api";
import TranscriptPanel from "./TranscriptPanel";
import ChatPanel from "./ChatPanel";
import SearchPanel from "./SearchPanel";
import VisualSearchPanel from "./VisualSearchPanel";

function formatTime(seconds) {
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export default function VideoWorkspace({ video }) {
  const videoRef = useRef(null);
  const [tab, setTab] = useState("chat");

  const seekTo = (seconds) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play().catch(() => {});
    }
  };

  return (
    <div className="workspace">
      <div className="video-stage">
        <video ref={videoRef} src={api.videoFileUrl(video.video_id)} controls className="video-player" />
      </div>

      <div className="workspace-columns">
        <TranscriptPanel video={video} onSeek={seekTo} formatTime={formatTime} />

        <div className="panel assistant-panel">
          <div className="tab-row">
            <button className={tab === "chat" ? "tab active" : "tab"} onClick={() => setTab("chat")}>
              Ask
            </button>
            <button className={tab === "search" ? "tab active" : "tab"} onClick={() => setTab("search")}>
              Search transcript
            </button>
            <button className={tab === "visual" ? "tab active" : "tab"} onClick={() => setTab("visual")}>
              Search visually
            </button>
          </div>
          {tab === "chat" && <ChatPanel video={video} onSeek={seekTo} formatTime={formatTime} />}
          {tab === "search" && <SearchPanel video={video} onSeek={seekTo} formatTime={formatTime} />}
          {tab === "visual" && <VisualSearchPanel video={video} onSeek={seekTo} formatTime={formatTime} />}
        </div>
      </div>
    </div>
  );
}
