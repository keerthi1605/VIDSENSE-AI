import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import FormattedAnswer from "./FormattedAnswer";

export default function ChatPanel({ video, onSeek, formatTime }) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]); // {query, answer, sources, hasContext, error}
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const messagesRef = useRef(null);

  // Keep the newest answer (and its citation chips) in view.
  useEffect(() => {
    const el = messagesRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  // A CPU-only language model takes 15-30 s per answer; show that it is working.
  useEffect(() => {
    if (!busy) return undefined;
    setElapsed(0);
    const timer = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [busy]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const q = query.trim();
    if (!q || busy) return;
    setBusy(true);
    setQuery("");
    try {
      const response = await api.chat(q, video.video_id);
      setMessages((prev) => [
        ...prev,
        {
          query: q,
          answer: response.answer,
          sources: response.sources,
          hasContext: response.has_sufficient_context,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [...prev, { query: q, error: err.message }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-messages" ref={messagesRef}>
        {messages.length === 0 && (
          <p className="hint">Ask a question about this video — answers cite exact timestamps.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className="chat-exchange">
            <div className="chat-bubble chat-user">{m.query}</div>
            {m.error ? (
              <div className="chat-bubble chat-error">{m.error}</div>
            ) : (
              <div className={`chat-bubble chat-assistant ${!m.hasContext ? "chat-no-context" : ""}`}>
                <FormattedAnswer text={m.answer} />
                {m.sources?.length > 0 && (
                  <div className="chat-sources">
                    {m.sources.map((s, j) => {
                      // Land on the best segment inside the cited chunk when the
                      // backend found one; otherwise the chunk start.
                      const target = s.jump_time ?? s.start_time;
                      return (
                        <button
                          key={j}
                          className="source-chip"
                          title={`Cited passage ${formatTime(s.start_time)}–${formatTime(s.end_time)}`}
                          onClick={() => onSeek(target)}
                        >
                          ▶ {formatTime(target)} (score {s.score.toFixed(2)})
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {busy && (
          <div className="chat-bubble chat-assistant chat-thinking">
            Thinking… {elapsed > 0 && <span className="thinking-timer">{elapsed} s</span>}
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="chat-input-row">
        <input
          type="text"
          placeholder="e.g. How does round robin scheduling work?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !query.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}
