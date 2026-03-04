import { useState, useRef } from "react";
import { analyzeImage, analyzeBrainMRI } from "../api/api";

/* ---------- pre-upload validation ---------- */
const MAX_SIZE_MB = 10;
const MIN_SIZE_KB = 2;
function validateFile(f) {
  if (!["image/jpeg", "image/png"].includes(f.type))
    return "Only JPG and PNG files are accepted.";
  if (f.size > MAX_SIZE_MB * 1024 * 1024)
    return `File is too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Maximum is ${MAX_SIZE_MB} MB.`;
  if (f.size < MIN_SIZE_KB * 1024)
    return `File is too small (${f.size} bytes). This is unlikely to be a valid scan.`;
  return null;
}

/* ---------- Simulation mode card ---------- */
function SimulationCard() {
  return (
    <div className="simulation-mode-card" role="alert">
      <div className="sim-card-header">
        <span className="sim-card-icon">⚙️</span>
        <div>
          <div className="sim-card-title">Simulation Mode — No Real Prediction</div>
          <div className="sim-card-sub">AI model weights are not loaded on this server</div>
        </div>
      </div>
      <div className="sim-card-body">
        <p>
          The backend is running in <strong>simulation mode</strong> because the trained
          model weight files (<code>.pt</code>) are not present on this deployment.
          Results are <strong>randomly generated</strong> from a hash of your file —
          they are completely meaningless and must not be used for any purpose.
        </p>
        <div className="sim-card-note">
          <strong>For developers:</strong> Place <code>multi_cancer_classifier.pt</code> and{" "}
          <code>mri_classifier.pt</code> inside <code>backend/models/</code> and restart
          the server to enable real inference.
        </div>
      </div>
    </div>
  );
}

/* ---------- Invalid image card ---------- */
function InvalidCard({ results }) {
  // Collect unique warnings from all models that flagged invalid
  const warnings = results
    .filter(r => r && r.valid_medical_image === false && r.warning)
    .map(r => r.warning);
  const unique = [...new Set(warnings)];

  return (
    <div className="invalid-image-card" role="alert">
      <div className="invalid-card-header">
        <span className="invalid-card-icon">🚫</span>
        <div>
          <div className="invalid-card-title">Not a Valid Medical Image</div>
          <div className="invalid-card-sub">Upload a medical scan for reliable results</div>
        </div>
      </div>
      <div className="invalid-card-body">
        {unique.map((w, i) => (
          <p key={i} className="invalid-card-warning">{w}</p>
        ))}
        <div className="invalid-card-tips">
          <strong>Accepted image types:</strong>
          <ul>
            <li>🧠 Brain MRI — T1/T2 weighted axial or coronal slices</li>
            <li>🔬 Histopathology slides — tissue microscopy (H&amp;E stained)</li>
            <li>🩸 Bone marrow smears for blood cancer detection</li>
            <li>🫁 Lung / colon tissue microscopy slides</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

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
  const isInvalid = result.valid_medical_image === false;

  return (
    <div className={`finding-card${isInvalid ? " finding-card--invalid" : ""}`}>
      {/* top row: icon + type badge */}
      <div className="finding-top">
        <span className="finding-icon">{isInvalid ? "⚠️" : neg ? "✅" : "⚠️"}</span>
        <div className="finding-info">
          <span className="finding-source">{label}</span>
          <span className={"cancer-type-badge cancer-type-badge--" + slug}>
            {result.cancer_type || "Unknown"}
          </span>
        </div>
        {result.simulated && (
          <span className="badge-simulated">Simulated</span>
        )}
        {isInvalid && (
          <span className="badge-invalid">Low Confidence</span>
        )}
      </div>

      {/* invalid warning */}
      {isInvalid && result.warning && (
        <div className="finding-invalid-warning">{result.warning}</div>
      )}

      {/* subclass */}
      <div className="finding-row">
        <span className="finding-lbl">Subclass</span>
        <span className="finding-val" style={isInvalid ? { opacity: 0.5 } : {}}>
          {fmt(result.prediction)}
          {isInvalid && <span className="finding-unreliable"> (unreliable)</span>}
        </span>
      </div>

      {/* description (MRI model) */}
      {result.description && (
        <div className="finding-desc" style={isInvalid ? { opacity: 0.5 } : {}}>
          {result.description}
        </div>
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
  const [fileError,   setFileError]  = useState(null);
  const [apiError,    setApiError]   = useState(null);
  const inputRef = useRef();

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    // Reset prior state
    if (preview) URL.revokeObjectURL(preview);
    setMulti(null); setMri(null); setApiError(null);

    const err = validateFile(f);
    if (err) {
      setFileError(err);
      setFile(null); setPreview(null);
      if (inputRef.current) inputRef.current.value = "";
      return;
    }
    setFileError(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setMulti(null); setMri(null); setApiError(null);

    const [r1, r2] = await Promise.allSettled([
      analyzeImage(file),
      analyzeBrainMRI(file),
    ]);
    if (r1.status === "fulfilled") setMulti(r1.value);
    if (r2.status === "fulfilled") setMri(r2.value);

    // Surface API errors
    const errs = [r1, r2]
      .filter(r => r.status === "rejected")
      .map(r => r.reason?.response?.data?.detail || r.reason?.message || "Analysis failed");
    if (errs.length === 2) setApiError(errs[0]);

    setLoading(false);
  }

  function handleReset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null);
    setMulti(null); setMri(null);
    setFileError(null); setApiError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  // Classify each result
  const multiSimulated = multiResult?.simulated === true;
  const mriSimulated   = mriResult?.simulated   === true;
  const bothSimulated  = multiResult && mriResult && multiSimulated && mriSimulated;
  const anySimulated   = multiSimulated || mriSimulated;

  const multiInvalid = multiResult && !multiSimulated && multiResult.valid_medical_image === false;
  const mriInvalid   = mriResult   && !mriSimulated   && mriResult.valid_medical_image   === false;
  const bothInvalid  = multiResult && mriResult && multiInvalid && mriInvalid;
  const anyValid     = (multiResult && !multiInvalid && !multiSimulated) ||
                       (mriResult   && !mriInvalid   && !mriSimulated);

  // Only surface positive findings from VALID, non-simulated results
  const concerning = [
    multiResult && !multiInvalid && !multiSimulated && !isNeg(multiResult.prediction)
      ? { result: multiResult, label: "Multi-Cancer Scan" } : null,
    mriResult   && !mriInvalid   && !mriSimulated   && !isNeg(mriResult.prediction)
      ? { result: mriResult,   label: "Brain MRI Scan"   } : null,
  ].filter(Boolean);

  const analysisRan = multiResult || mriResult;
  const allClear    = analysisRan && anyValid && concerning.length === 0;

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

      {/* Pre-upload file validation error */}
      {fileError && (
        <div className="alert alert-error" role="alert">
          <span className="alert-body"><strong>Invalid file: </strong>{fileError}</span>
          <button className="alert-dismiss" onClick={() => setFileError(null)}>×</button>
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

      {/* API error (both models failed entirely) */}
      {apiError && !loading && (
        <div className="alert alert-error" role="alert">
          <span className="alert-body"><strong>Error: </strong>{apiError}</span>
          <button className="alert-dismiss" onClick={() => setApiError(null)}>×</button>
        </div>
      )}

      {analysisRan && !loading && (
        <div className="results-section" aria-live="polite">
          <p className="result-heading">Analysis Results</p>

          {/* ── SIMULATION MODE (no model weights loaded) ─────────────── */}
          {bothSimulated && (
            <SimulationCard />
          )}

          {/* ── Both real models rejected the image as non-medical ─────── */}
          {!bothSimulated && bothInvalid && (
            <InvalidCard results={[multiResult, mriResult]} />
          )}

          {/* ── At least one real model accepted it → normal result flow ─ */}
          {!bothSimulated && !bothInvalid && (
            <>
              {/* Partial simulation notice */}
              {anySimulated && (
                <div className="partial-invalid-notice">
                  ⚙️ <strong>Note:</strong> One or more models ran in simulation mode.
                  Results below come only from the non-simulated model.
                </div>
              )}

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

              {/* Inline notice if ONE model rejected as invalid (real inference) */}
              {multiInvalid && !mriInvalid && (
                <div className="partial-invalid-notice">
                  ⚠️ <strong>Multi-Cancer model:</strong> {multiResult.warning}
                </div>
              )}
              {mriInvalid && !multiInvalid && (
                <div className="partial-invalid-notice">
                  ⚠️ <strong>Brain MRI model:</strong> {mriResult.warning}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}
