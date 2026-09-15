import { useEffect, useState } from "react";
import { api } from "../api";

// Two independent pipelines, matching the project's own phase split:
// text (Phases 1-4: Whisper -> chunk -> embed -> index) and visual
// (Phase 5: frames -> CLIP -> index). Each step maps to exactly one
// backend endpoint -- this component is deliberately just a thin
// sequencer + status board over calls that already exist.
const TEXT_STEPS = [
  { key: "transcribe", label: "Transcribe (Whisper)", run: (id) => api.transcribe(id) },
  { key: "chunk", label: "Chunk transcript", run: (id) => api.chunk(id) },
  { key: "embed", label: "Embed chunks (MiniLM)", run: (id) => api.embed(id) },
  { key: "index", label: "Index into vector DB", run: (id) => api.index(id) },
];

const VISUAL_STEPS = [
  { key: "frames", label: "Extract frames", run: (id) => api.extractFrames(id) },
  { key: "embedFrames", label: "Embed frames (CLIP)", run: (id) => api.embedFrames(id) },
  { key: "indexFrames", label: "Index frames into vector DB", run: (id) => api.indexFrames(id) },
];

function StepRow({ label, status, error }) {
  const icon = { pending: "○", running: "◐", done: "✓", error: "✕" }[status];
  return (
    <li className={`step-row step-${status}`}>
      <span className="step-icon">{icon}</span>
      <span className="step-label">{label}</span>
      {status === "error" && <span className="step-error">{error}</span>}
    </li>
  );
}

export default function PipelineStatus({ video, onUpdated }) {
  const [textStatus, setTextStatus] = useState({});
  const [visualStatus, setVisualStatus] = useState({});
  const [textRunning, setTextRunning] = useState(false);
  const [visualRunning, setVisualRunning] = useState(false);

  // Probe what's already done for this video (e.g. re-selecting a
  // previously fully-processed video shouldn't look unprocessed).
  useEffect(() => {
    let cancelled = false;
    async function probe() {
      const next = {};
      try {
        await api.getTranscript(video.video_id);
        next.transcribe = "done";
        try {
          await api.getChunks(video.video_id);
          next.chunk = "done";
          next.embed = "done"; // chunks existing implies embed+index were attempted together in this UI
          next.index = "done";
        } catch {
          /* not chunked yet */
        }
      } catch {
        /* not transcribed yet */
      }
      if (!cancelled) setTextStatus(next);

      const nextVisual = {};
      try {
        await api.getFrames(video.video_id);
        nextVisual.frames = "done";
        nextVisual.embedFrames = "done";
        nextVisual.indexFrames = "done";
      } catch {
        /* not frame-extracted yet */
      }
      if (!cancelled) setVisualStatus(nextVisual);
    }
    probe();
    return () => {
      cancelled = true;
    };
  }, [video.video_id]);

  async function runPipeline(steps, setStatus, setRunning) {
    setRunning(true);
    for (const step of steps) {
      setStatus((prev) => ({ ...prev, [step.key]: "running" }));
      try {
        await step.run(video.video_id);
        setStatus((prev) => ({ ...prev, [step.key]: "done" }));
      } catch (err) {
        setStatus((prev) => ({ ...prev, [step.key]: "error" }));
        setTextRunning(false);
        setVisualRunning(false);
        return; // stop the sequence on first failure
      }
    }
    setRunning(false);
    try {
      const refreshed = await api.getVideo(video.video_id);
      onUpdated(refreshed);
    } catch {
      /* non-fatal */
    }
  }

  const allTextDone = TEXT_STEPS.every((s) => textStatus[s.key] === "done");
  const allVisualDone = VISUAL_STEPS.every((s) => visualStatus[s.key] === "done");

  return (
    <div className="panel pipeline-panel">
      <div className="pipeline-columns">
        <div className="pipeline-column">
          <div className="pipeline-column-header">
            <h3>Text pipeline</h3>
            <button
              onClick={() => runPipeline(TEXT_STEPS, setTextStatus, setTextRunning)}
              disabled={textRunning || allTextDone}
            >
              {allTextDone ? "Done" : textRunning ? "Running…" : "Run"}
            </button>
          </div>
          <ul className="step-list">
            {TEXT_STEPS.map((s) => (
              <StepRow key={s.key} label={s.label} status={textStatus[s.key] || "pending"} />
            ))}
          </ul>
        </div>

        <div className="pipeline-column">
          <div className="pipeline-column-header">
            <h3>Visual pipeline</h3>
            <button
              onClick={() => runPipeline(VISUAL_STEPS, setVisualStatus, setVisualRunning)}
              disabled={visualRunning || allVisualDone}
            >
              {allVisualDone ? "Done" : visualRunning ? "Running…" : "Run"}
            </button>
          </div>
          <ul className="step-list">
            {VISUAL_STEPS.map((s) => (
              <StepRow key={s.key} label={s.label} status={visualStatus[s.key] || "pending"} />
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
