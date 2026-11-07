import { useEffect, useRef, useState } from "react";
import { api, imageUrl } from "./api.js";
import { createLatest } from "./latest.js";

// Slider ranges are the useful interactive span; the server validates the wider API range.
// Seed, colour σ and spatial scale are fixed server-side and not user parameters.
const CLUSTER_FIELDS = [
  { name: "k", label: "Max segments (k)", min: 2, max: 100, step: 1, integer: true },
  { name: "ncut_threshold", label: "Ncut threshold", min: 0.01, max: 0.5, step: 0.01 },
];
const GRAPH_FIELDS = [
  { name: "superpixels", label: "Superpixels", min: 200, max: 6000, step: 100, integer: true },
  { name: "blur_sigma", label: "Blur σ", min: 0, max: 5, step: 0.1 },
  { name: "max_dim", label: "Max side (px)", min: 200, max: 1600, step: 50, integer: true },
];
// The UI always uses recursive NJW; flat stays available through the API and CLI.
const DEFAULTS = { k: "80", ncut_threshold: "0.05", superpixels: "3000", blur_sigma: "1", max_dim: "800" };
const OVERLAYS = ["segmentation", "boundaries", "superpixels"];

function toRequest(form) {
  const params = { method: "recursive" };
  for (const f of [...CLUSTER_FIELDS, ...GRAPH_FIELDS]) {
    const text = String(form[f.name]).trim();
    const value = Number(text);
    if (text === "" || !Number.isFinite(value) || (f.integer && !Number.isInteger(value))) {
      throw new Error(`${f.label} must be ${f.integer ? "an integer" : "a number"}`);
    }
    params[f.name] = value;
  }
  return params;
}

const describe = (error) => (error?.code ? `${error.message} (${error.code})` : String(error?.message ?? error));
const ms = (x) => (x == null ? "—" : x < 10 ? `${x.toFixed(1)} ms` : `${Math.round(x)} ms`);
const sci = (x) => (x == null ? "—" : x.toExponential(1));
const shortId = (id) => id.slice(0, 8);
const when = (iso) => new Date(iso).toLocaleString();

function ResultId({ id }) {
  return (
    <code title={id} data-testid="result-id">
      {shortId(id)}
    </code>
  );
}

export default function App() {
  const [run, setRun] = useState(null); // uploaded image: {run_id, filename, width, height}
  const [form, setForm] = useState(DEFAULTS);
  // The result on screen. Only the latest segmentation request may replace it.
  const [current, setCurrent] = useState(null); // {result, requestMs, roundTripMs}
  const [busy, setBusy] = useState(null); // "upload" | "segment" | null
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [overlay, setOverlay] = useState("segmentation");
  const [savingId, setSavingId] = useState(null);
  const [clearing, setClearing] = useState(false);
  const [history, setHistory] = useState([]);
  const [viewing, setViewing] = useState(null); // a saved experiment opened from history

  const uploads = useRef(createLatest());
  const segments = useRef(createLatest());
  const lists = useRef(createLatest());

  async function refreshHistory() {
    try {
      const r = await lists.current.run((signal) => api.experiments(signal));
      if (r.current) setHistory(r.value.data.experiments);
    } catch (e) {
      setError(`Could not load saved experiments: ${describe(e)}`);
    }
  }

  useEffect(() => {
    refreshHistory();
  }, []);

  async function onFile(file) {
    if (!file) return;
    segments.current.invalidate(); // a result for the previous image must not appear later
    setCurrent(null);
    setError(null);
    setNotice(null);
    setBusy("upload");
    try {
      const r = await uploads.current.run((signal) => api.upload(file, signal));
      if (!r.current) return;
      setRun(r.value.data);
      setBusy(null);
    } catch (e) {
      setError(`Upload failed: ${describe(e)}`);
      setBusy(null);
    }
  }

  async function onSegment(event) {
    event.preventDefault();
    if (!run) return;
    let params;
    try {
      params = toRequest(form);
    } catch (e) {
      setError(e.message);
      return;
    }
    const runId = run.run_id;
    setError(null);
    setNotice(null);
    setBusy("segment");
    try {
      const r = await segments.current.run((signal) => api.segment(runId, params, signal));
      if (!r.current) return; // superseded by a newer request or a new upload
      setCurrent({ result: r.value.data, requestMs: r.value.meta.request_ms, roundTripMs: r.value.roundTripMs });
      setBusy(null);
    } catch (e) {
      setError(`Segmentation failed: ${describe(e)}`);
      setBusy(null);
    }
  }

  async function onSave() {
    const id = current?.result.result_id; // exactly the result on screen
    if (!id) return;
    setSavingId(id);
    try {
      const { data } = await api.save(id);
      if (data.result_id !== id) throw new Error(`server saved ${data.result_id}, expected ${id}`);
      setCurrent((c) => (c && c.result.result_id === id ? { ...c, result: { ...c.result, saved_at: data.saved_at } } : c));
      refreshHistory();
    } catch (e) {
      setError(`Save failed: ${describe(e)}`);
    } finally {
      setSavingId((s) => (s === id ? null : s));
    }
  }

  async function onClearHistory() {
    if (!window.confirm(`Permanently delete all ${history.length} saved experiments?`)) return;
    setClearing(true);
    setError(null);
    try {
      const { data } = await api.clearHistory();
      setViewing(null);
      // A saved result on screen no longer exists on the server.
      setCurrent((c) => (c?.result.saved_at ? null : c));
      setNotice(`Deleted ${data.deleted} saved experiment${data.deleted === 1 ? "" : "s"}.`);
      await refreshHistory();
    } catch (e) {
      setError(`Clear failed: ${describe(e)}`);
    } finally {
      setClearing(false);
    }
  }

  const set = (name) => (e) => setForm((f) => ({ ...f, [name]: e.target.value }));
  const slider = (f) => (
      <div className="slider" key={f.name}>
        <label htmlFor={`range-${f.name}`}>{f.label}</label>
        <input
          id={`range-${f.name}`}
          type="range"
          min={f.min}
          max={f.max}
          step={f.step}
          value={form[f.name]}
          onChange={set(f.name)}
        />
        <input
          name={f.name}
          className="value"
          inputMode="decimal"
          aria-label={`${f.label} value`}
          value={form[f.name]}
          onChange={set(f.name)}
        />
      </div>
    );

  return (
    <div className="app">
      <header>
        <h1>Spectral image segmentation</h1>
      </header>

      <aside>
        <section>
          <h2>Image</h2>
          <input type="file" accept="image/*" aria-label="Image file" onChange={(e) => onFile(e.target.files?.[0])} />
          {run && (
            <p className="muted" data-testid="run-info">
              {run.filename} · {run.width}×{run.height}
            </p>
          )}
        </section>

        <form onSubmit={onSegment}>
          <h2>Parameters</h2>
          <fieldset>
            <legend>Clustering</legend>
            {CLUSTER_FIELDS.map(slider)}
          </fieldset>
          <fieldset>
            <legend>Superpixel graph</legend>
            {GRAPH_FIELDS.map(slider)}
          </fieldset>
          <button type="submit" disabled={!run || busy === "upload"}>
            {busy === "segment" ? "Segmenting…" : "Segment"}
          </button>
          <button type="button" className="secondary" onClick={() => setForm(DEFAULTS)}>
            Reset defaults
          </button>
        </form>
      </aside>

      <main>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {notice && <p className="notice">{notice}</p>}
        {busy === "upload" && <p className="muted">Uploading…</p>}
        {!current && busy !== "upload" && (
          <p className="muted">{run ? "Choose parameters and press Segment." : "Upload an image to start."}</p>
        )}
        {current && (
          <section
            className={`result ${busy === "segment" ? "stale" : ""}`}
            data-testid="current-result"
            data-result-id={current.result.result_id}
          >
            <div className="result-head">
              <h2>
                Current result <ResultId id={current.result.result_id} />
              </h2>
              {busy === "segment" && <span className="muted">computing a new result — showing the previous one</span>}
              {current.result.saved_at ? (
                <span className="saved" data-testid="saved-badge">
                  Saved {when(current.result.saved_at)}
                </span>
              ) : (
                <button onClick={onSave} disabled={savingId === current.result.result_id} data-testid="save">
                  {savingId === current.result.result_id ? "Saving…" : `Save result ${shortId(current.result.result_id)}`}
                </button>
              )}
            </div>
            <Comparison result={current.result} overlay={overlay} setOverlay={setOverlay} />
            <Metrics result={current.result} requestMs={current.requestMs} roundTripMs={current.roundTripMs} />
          </section>
        )}

        <section>
          <div className="result-head">
            <h2>Saved experiments</h2>
            {history.length > 0 && (
              <button className="danger" onClick={onClearHistory} disabled={clearing} data-testid="clear-history">
                {clearing ? "Clearing…" : "Clear history"}
              </button>
            )}
          </div>
          {history.length === 0 ? (
            <p className="muted">Nothing saved yet.</p>
          ) : (
            <table data-testid="history">
              <thead>
                <tr>
                  <th>Saved</th>
                  <th>Result</th>
                  <th>Superpixels</th>
                  <th>Segments</th>
                  <th>Engine time</th>
                </tr>
              </thead>
              <tbody>
                {history.map((e) => (
                  <tr
                    key={e.result_id}
                    className={viewing?.result_id === e.result_id ? "selected" : ""}
                    onClick={() => setViewing(e)}
                    data-testid={`history-${e.result_id}`}
                  >
                    <td>{when(e.saved_at)}</td>
                    <td>
                      <ResultId id={e.result_id} />
                    </td>
                    <td>{e.params.superpixels}</td>
                    <td>{e.metrics.segments}</td>
                    <td>{ms(e.metrics.compute_ms.total)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {viewing && (
          <section className="viewing" data-testid="viewing">
            <div className="result-head">
              <h2>
                Saved experiment <ResultId id={viewing.result_id} />
              </h2>
              <span className="muted">saved {when(viewing.saved_at)} · read-only</span>
              <button className="secondary" onClick={() => setViewing(null)}>
                Close
              </button>
            </div>
            <ParamsSummary params={viewing.params} />
            <Comparison result={viewing} overlay={overlay} setOverlay={setOverlay} />
            <Metrics result={viewing} />
          </section>
        )}
      </main>
    </div>
  );
}

function Comparison({ result, overlay, setOverlay }) {
  return (
    <>
      <div className="overlay-picker" role="group" aria-label="Overlay">
        {OVERLAYS.map((o) => (
          <button key={o} className={o === overlay ? "active" : "secondary"} onClick={() => setOverlay(o)}>
            {o}
          </button>
        ))}
      </div>
      <div className="compare">
        <figure>
          <img src={imageUrl(result.result_id, "original")} alt="original" />
          <figcaption>Original (working size)</figcaption>
        </figure>
        <figure>
          <img src={imageUrl(result.result_id, overlay)} alt={overlay} data-testid="overlay-image" />
          <figcaption>{overlay}</figcaption>
        </figure>
      </div>
    </>
  );
}

function ParamsSummary({ params }) {
  const keys = ["k", "superpixels", "ncut_threshold", "blur_sigma", "max_dim"];
  return <p className="muted">{keys.map((k) => `${k}=${params[k]}`).join(" · ")}</p>;
}

// requestMs/roundTripMs exist only for a result computed in this session; they describe
// the HTTP request, not the experiment, and are never stored.
function Metrics({ result, requestMs, roundTripMs }) {
  const m = result.metrics;
  const c = m.compute_ms;
  return (
    <div className="metrics" data-testid="metrics">
      <dl>
        <dt>Segments</dt>
        <dd data-testid="segments">{m.segments}</dd>
        <dt>Superpixels</dt>
        <dd>{m.superpixels}</dd>
        <dt>Ncut</dt>
        <dd>{m.spectral.ncut == null ? "—" : m.spectral.ncut.toFixed(3)}</dd>
      </dl>
      <dl>
        <dt>Max residual</dt>
        <dd>{sci(m.spectral.max_residual)}</dd>
        <dt>Orthogonality</dt>
        <dd>{sci(m.spectral.orthogonality_error)}</dd>
        <dt>Spectral gap</dt>
        <dd>{sci(m.spectral.spectral_gap)}</dd>
      </dl>
      <dl>
        <dt>Engine time (this result)</dt>
        <dd data-testid="engine-total">{ms(c.total)}</dd>
        <dt>eigen</dt>
        <dd>{ms(c.eigen)}</dd>
        <dt>k-means</dt>
        <dd>{ms(c.kmeans)}</dd>
      </dl>
      {requestMs !== undefined && (
        <dl>
          <dt>HTTP request (server)</dt>
          <dd data-testid="request-ms">{ms(requestMs)}</dd>
          <dt>Round trip (browser)</dt>
          <dd>{ms(roundTripMs)}</dd>
        </dl>
      )}
    </div>
  );
}
