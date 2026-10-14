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
  { key: "frames", label: "Extract frames (SSIM scenes)", run: (id) => api.extractFrames(id) },
  { key: "embedFrames", label: "Embed frames (CLIP)", run: (id) => api.embedFrames(id) },
  { key: "indexFrames", label: "Index frames into vector DB", run: (id) => api.indexFrames(id) },
  {
    // Chunking is aligned to the scene changes found above, so if the
    // transcript was already chunked (before frames existed) it has to be
    // redone. With no transcript yet there is nothing to align: the chunk
    // step will pick the frames up whenever the text pipeline runs.
    key: "align",
    label: "Align text chunks to scenes",
    run: async (id) => {
      try {
        await api.getTranscript(id);
      } catch {
        return;
      }
      await api.chunk(id);
      await api.embed(id);
      await api.index(id);
    },
  },
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
      let chunkAlignment = null; // null = not chunked yet
      try {
        await api.getTranscript(video.video_id);
        next.transcribe = "done";
        try {
          const chunkSet = await api.getChunks(video.video_id);
          chunkAlignment = chunkSet.alignment;
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
        // Aligned if chunks already follow the scenes, or if there are no
        // chunks yet (they will be aligned when created).
        if (chunkAlignment === null || chunkAlignment === "scene_aligned") nextVisual.align = "done";
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

  async function runPipeline(steps, currentStatus, setStatus, setRunning) {
    setRunning(true);
    for (const step of steps) {
      if (currentStatus[step.key] === "done") continue; // don't redo finished (slow) steps
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
              onClick={() => runPipeline(TEXT_STEPS, textStatus, setTextStatus, setTextRunning)}
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
              onClick={() => runPipeline(VISUAL_STEPS, visualStatus, setVisualStatus, setVisualRunning)}
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
