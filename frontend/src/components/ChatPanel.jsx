import { useState } from "react";
import { api } from "../api";

export default function ChatPanel({ video, onSeek, formatTime }) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]); // {query, answer, sources, hasContext, error}
  const [busy, setBusy] = useState(false);

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
      <div className="chat-messages">
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
                <p>{m.answer}</p>
                {m.sources?.length > 0 && (
                  <div className="chat-sources">
                    {m.sources.map((s, j) => (
                      <button key={j} className="source-chip" onClick={() => onSeek(s.start_time)}>
                        ▶ {formatTime(s.start_time)} (score {s.score.toFixed(2)})
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {busy && <div className="chat-bubble chat-assistant chat-thinking">Thinking…</div>}
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
