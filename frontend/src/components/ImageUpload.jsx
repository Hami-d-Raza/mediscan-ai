import { useState, useRef } from "react";
import { analyzeImage, analyzeBrainMRI } from "../api/api";

/* ---------- display name map for all subclasses ---------- */
const SUBCLASS_NAMES = {
  all_benign:       "ALL – Benign",
  all_early:        "ALL – Early Stage",
  all_pre:          "ALL – Pre Stage",
  all_pro:          "ALL – Pro Stage",
  brain_glioma:     "Glioma",
  brain_menin:      "Meningioma",
  brain_tumor:      "Brain Tumor",
  breast_benign:    "Breast – Benign",
  breast_malignant: "Breast – Malignant",
  cervix_dyk:       "Cervical – Dyskeratotic",
  cervix_koc:       "Cervical – Koilocytotic",
  cervix_mep:       "Cervical – Metaplastic",
  cervix_pab:       "Cervical – Parabasal",
  cervix_sfi:       "Cervical – Superficial",
  colon_aca:        "Colon Adenocarcinoma",
  colon_bnt:        "Colon – Benign",
  lung_aca:         "Lung Adenocarcinoma",
  lung_bnt:         "Lung – Benign",
  lung_scc:         "Lung Squamous Cell Carcinoma",
  kidney_normal:    "Kidney – Normal",
  kidney_tumor:     "Kidney Tumor",
  lymph_cll:        "CLL (Chronic Lymphocytic Leukemia)",
  lymph_fl:         "FL (Follicular Lymphoma)",
  lymph_mcl:        "MCL (Mantle Cell Lymphoma)",
  oral_normal:      "Oral – Normal",
  oral_scc:         "Oral Squamous Cell Carcinoma",
};

function fmt(raw) { return SUBCLASS_NAMES[raw] || raw; }

function isNeg(pred) {
  const p = (pred || "").toLowerCase();
  return p.includes("normal") || p.includes("benign") ||
         p.includes("no tumor") || p.includes("no_tumor");
}

function color(pct) {
  if (pct >= 75) return "#16a34a";
  if (pct >= 50) return "#d97706";
  return "#dc2626";
}

/* ---------- single finding card (no model label shown) ---------- */
function FindingCard({ result, label }) {
  if (!result) return null;
  const pct   = +(result.confidence * 100).toFixed(1);
  const clr   = color(pct);
  const neg   = isNeg(result.prediction);
  const slug  = (result.cancer_type || "")
                  .toLowerCase().replace(/[^a-z]/g, "-")
                  .replace(/-+/g, "-").replace(/^-|-$/g, "");

  return (
    <div className="finding-card">
      {/* top row: icon + type badge */}
      <div className="finding-top">
        <span className="finding-icon">{neg ? "✅" : "⚠️"}</span>
        <div className="finding-info">
          <span className="finding-source">{label}</span>
          <span className={"cancer-type-badge cancer-type-badge--" + slug}>
            {result.cancer_type || "Unknown"}
          </span>
        </div>
        {result.simulated && (
          <span className="badge-simulated">Simulated</span>
        )}
      </div>

      {/* subclass */}
      <div className="finding-row">
        <span className="finding-lbl">Subclass</span>
        <span className="finding-val">{fmt(result.prediction)}</span>
      </div>

      {/* description (MRI model) */}
      {result.description && (
        <div className="finding-desc">{result.description}</div>
      )}

      {/* confidence bar */}
      <div className="confidence-block">
        <div className="confidence-header">
          <span>Confidence</span>
          <span className="confidence-pct" style={{ color: clr }}>{pct}%</span>
        </div>
        <div className="confidence-track">
          <div className="confidence-fill" style={{ width: pct + "%", background: clr }} />
        </div>
      </div>

      <div className="result-meta">
        <span>{result.filename}</span>
        <span>Class #{result.class_index}</span>
      </div>
    </div>
  );
}

/* ============================================================ */
export default function ImageUpload() {
  const [file,        setFile]       = useState(null);
  const [preview,     setPreview]    = useState(null);
  const [multiResult, setMulti]      = useState(null);
  const [mriResult,   setMri]        = useState(null);
  const [loading,     setLoading]    = useState(false);
  const inputRef = useRef();

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setMulti(null); setMri(null);
    setPreview(URL.createObjectURL(f));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setMulti(null); setMri(null);

    const [r1, r2] = await Promise.allSettled([
      analyzeImage(file),
      analyzeBrainMRI(file),
    ]);
    if (r1.status === "fulfilled") setMulti(r1.value);
    if (r2.status === "fulfilled") setMri(r2.value);

    setLoading(false);
  }

  function handleReset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null);
    setMulti(null); setMri(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  // Only surface results where a problem was detected
  const concerning = [
    multiResult && !isNeg(multiResult.prediction) ? { result: multiResult, label: "Multi-Cancer Scan" } : null,
    mriResult   && !isNeg(mriResult.prediction)   ? { result: mriResult,   label: "Brain MRI Scan"   } : null,
  ].filter(Boolean);

  const analysisRan  = multiResult || mriResult;
  const allClear     = analysisRan && concerning.length === 0;

  return (
    <section className="card">
      {loading && (
        <div className="card-overlay" aria-live="polite">
          <span className="spinner-lg" />
          <p>Analyzing with AI models…</p>
        </div>
      )}

      <div className="card-header">
        <div className="card-icon card-icon--image">🔬</div>
        <div className="card-header-text">
          <h2 className="card-title">Medical Image Analysis</h2>
          <p className="card-desc">
            Upload a medical image and our AI identifies the cancer type and subclass
            across <strong>8 cancer categories</strong> and <strong>26 subclasses</strong>,
            including dedicated <strong>Brain MRI</strong> detection.
          </p>
        </div>
      </div>

      {!file ? (
        <div className="dropzone">
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleFileChange}
            aria-label="Choose medical image"
          />
          <span className="dropzone-icon">🧻</span>
          <span className="dropzone-main">Drop your medical image here</span>
          <span className="dropzone-sub">or click to browse files</span>
          <span className="dropzone-hint">JPG, PNG — MRI, histology, microscopy</span>
        </div>
      ) : (
        <div className="file-selected">
          <span className="file-selected-icon">🖼️</span>
          <div className="file-selected-info">
            <div className="file-selected-name">{file.name}</div>
            <div className="file-selected-size">{(file.size / 1024).toFixed(1)} KB</div>
          </div>
          <button type="button" className="btn-clear" onClick={handleReset}>Remove</button>
        </div>
      )}

      {preview && (
        <div className="preview">
          <img src={preview} alt="Medical image preview" />
        </div>
      )}

      <button className="btn-primary" onClick={handleSubmit} disabled={!file || loading}>
        {loading ? <><span className="spinner" />Analyzing…</> : "Analyze Image"}
      </button>

      {analysisRan && !loading && (
        <div className="results-section" aria-live="polite">
          <p className="result-heading">Analysis Results</p>

          {allClear ? (
            <div className="finding-card finding-card--clear">
              <div className="finding-top">
                <span className="finding-icon">✅</span>
                <div className="finding-info">
                  <span className="finding-val" style={{ fontSize: "1rem" }}>No Abnormalities Detected</span>
                  <span className="finding-lbl">Both models returned normal / benign results</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="findings-list">
              {concerning.map(({ result, label }) => (
                <FindingCard key={label} result={result} label={label} />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
