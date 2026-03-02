import { useState, useRef } from "react";
import { analyzeReport } from "../api/api";

// ---------------------------------------------------------------------------
// Configuration: the three entity categories returned by the NLP backend.
// Each entry controls:  the response key, the heading label, the CSS colour
// modifier, and the placeholder text when the list is empty.
// Adding a new category (e.g. "procedures") only requires adding a row here.
// ---------------------------------------------------------------------------
const ENTITY_CATEGORIES = [
  {
    key: "diseases",
    label: "Diseases",
    modifier: "diseases",
    emptyText: "No diseases detected",
  },
  {
    key: "symptoms",
    label: "Symptoms",
    modifier: "symptoms",
    emptyText: "No symptoms detected",
  },
  {
    key: "medications",
    label: "Medications",
    modifier: "medications",
    emptyText: "No medications detected",
  },
  {
    key: "other",
    label: "MRI Findings",
    modifier: "findings",
    emptyText: "No imaging findings detected",
  },
];

// ---------------------------------------------------------------------------
// Helper: extract a user-readable error message from an axios error.
// FastAPI validation errors (422) arrive as an array under `detail`.
// All other errors arrive as a plain string under `detail`.
// ---------------------------------------------------------------------------
function parseError(err) {
  const detail = err.response?.data?.detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join(", ");
  return detail || err.message || "Report analysis failed. Please try again.";
}

// ===========================================================================
// ReportUpload
// ===========================================================================
// Allows the user to upload a PDF or plain-text medical report, then POSTs
// it to /analyze-report.  Displays the extracted medical entities
// (diseases, symptoms, medications) in colour-coded tag sections.
// ===========================================================================
export default function ReportUpload() {
  // ── State ──────────────────────────────────────────────────────────────
  const [file, setFile]       = useState(null);   // selected File object
  const [result, setResult]   = useState(null);   // API response payload
  const [error, setError]     = useState(null);   // human-readable error string
  const [loading, setLoading] = useState(false);  // true while request is in-flight

  // Ref lets us programmatically clear the native <input> value on reset
  const inputRef = useRef();

  // ── Handlers ───────────────────────────────────────────────────────────

  // Stores the newly picked file and clears any previous result / error.
  function handleFileChange(e) {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setResult(null);
    setError(null);
  }

  // Sends the report to the backend and stores the entity extraction result.
  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const data = await analyzeReport(file);  // see api/api.js
      setResult(data);
    } catch (err) {
      setError(parseError(err));
    } finally {
      setLoading(false);
    }
  }

  // Resets the card to its empty initial state.
  function handleReset() {
    setFile(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <section className="card">
      {loading && (
        <div className="card-overlay" aria-live="polite">
          <span className="spinner-lg" />
          <p>Extracting medical entities…</p>
        </div>
      )}

      {/* Header */}
      <div className="card-header">
        <div className="card-icon card-icon--report">📋</div>
        <div className="card-header-text">
          <h2 className="card-title">MRI Report Analysis</h2>
          <p className="card-desc">
            Upload an MRI report (PDF or TXT) to extract diseases, symptoms,
            medications, and imaging findings using biomedical NLP.
          </p>
        </div>
      </div>

      {/* Dropzone or selected file */}
      {!file ? (
        <div className="dropzone">
          <input ref={inputRef} type="file" accept=".pdf,.txt,text/plain,application/pdf" onChange={handleFileChange} aria-label="Choose medical report" />
          <span className="dropzone-icon">📄</span>
          <span className="dropzone-main">Drop your medical report here</span>
          <span className="dropzone-sub">or click to browse files</span>
          <span className="dropzone-hint">Supports PDF, TXT</span>
        </div>
      ) : (
        <div className="file-selected">
          <span className="file-selected-icon">📄</span>
          <div className="file-selected-info">
            <div className="file-selected-name">{file.name}</div>
            <div className="file-selected-size">
              {file.size >= 1024 * 1024 ? `${(file.size/(1024*1024)).toFixed(1)} MB` : `${(file.size/1024).toFixed(1)} KB`}
            </div>
          </div>
          <button type="button" className="btn-clear" onClick={handleReset}>Remove</button>
        </div>
      )}

      <button className="btn-primary" onClick={handleSubmit} disabled={!file || loading}>
        {loading ? <><span className="spinner" />Analyzing…</> : "Analyze Report"}
      </button>

      {error && (
        <div className="alert alert-error" role="alert">
          <span className="alert-body"><strong>Error: </strong>{error}</span>
          <button className="alert-dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}

      {result && !loading && (
        <div className="result" aria-live="polite">
          <p className="result-heading">Extracted Entities</p>

          <div className="result-meta">
            <span>File: {result.filename}</span>
            <span>{result.char_count?.toLocaleString()} chars</span>
            {result.simulated && <span className="badge-simulated">⚠️ Simulated</span>}
          </div>

          <div className="entity-sections">
            {ENTITY_CATEGORIES.map(({ key, label, modifier, emptyText }) => {
              const items = result.entities?.[key] ?? [];
              return (
                <div key={key} className={`entity-section entity-section--${modifier}`}>
                  <div className="entity-section-header">
                    <span>{label}</span>
                    <span className="entity-section-count">{items.length}</span>
                  </div>
                  {items.length > 0 ? (
                    <ul className="entity-tag-list">
                      {items.map((item, i) => <li key={i} className="entity-tag">{item}</li>)}
                    </ul>
                  ) : (
                    <p className="entity-empty">{emptyText}</p>
                  )}
                </div>
              );
            })}
          </div>

          {/* Suggested Medications */}
          {result.suggested_medications?.length > 0 && (
            <div className="entity-section entity-section--suggested">
              <div className="entity-section-header">
                <span>Suggested Medications</span>
                <span className="entity-section-count">{result.suggested_medications.length}</span>
              </div>
              <div className="disclaimer-banner">
                ⚠️ <strong>Disclaimer:</strong> These are general informational suggestions based on detected conditions only.
                <strong> Always consult a qualified physician</strong> before taking any medication.
                Do not self-medicate.
              </div>
              <div className="suggestion-list">
                {result.suggested_medications.map((s, i) => (
                  <div key={i} className="suggestion-row">
                    <div className="suggestion-drug">{s.drug}</div>
                    <div className="suggestion-meta">
                      <span className="suggestion-note">{s.note}</span>
                      <span className="suggestion-condition">For: {s.for_condition}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.summary && (
            <div className="summary">
              <span className="summary-label">Summary</span>
              <p>{result.summary}</p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
