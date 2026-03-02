import { Link } from "react-router-dom";
import { useState, useEffect } from "react";

/* ── Animated counter hook ── */
function useCounter(target, duration = 1400) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = Math.max(1, Math.ceil(target / (duration / 16)));
    const id = setInterval(() => {
      start += step;
      if (start >= target) { setCount(target); clearInterval(id); }
      else setCount(start);
    }, 16);
    return () => clearInterval(id);
  }, [target, duration]);
  return count;
}

function StatCounter({ value, suffix = "", label }) {
  const n = useCounter(value);
  return (
    <div className="hero-stat-block">
      <span className="hero-stat-number">{n.toLocaleString()}{suffix}</span>
      <span className="hero-stat-label">{label}</span>
    </div>
  );
}

const STEPS = [
  { num: "01", title: "Upload", desc: "Choose a medical scan (MRI, histopathology) or upload a clinical report (PDF / TXT).", icon: "upload" },
  { num: "02", title: "AI Analysis", desc: "Our deep-learning models process the input in under 2 seconds — no GPU required.", icon: "brain" },
  { num: "03", title: "Results", desc: "Get cancer type, subclass, confidence score, and extracted medical entities instantly.", icon: "chart" },
];

const MODELS = [
  {
    badge: "Multi-Cancer",
    badgeMod: "multi",
    name: "26-Class Histopathology Classifier",
    desc: "Trained on 135K+ images with MobileNetV2. Classifies across 8 cancer types and 26 subclasses from histopathology slides.",
    tags: ["MobileNetV2", "135K Images", "26 Classes"],
    cta: "Run Classifier",
    icon: "microscope",
  },
  {
    badge: "Brain MRI",
    badgeMod: "brain",
    name: "Brain MRI Tumor Classifier",
    desc: "Specialized model for brain MRI scans. Detects Glioma, Meningioma, Pituitary tumours, and healthy tissue with high confidence.",
    tags: ["Custom CNN", "7K Scans", "4 Classes"],
    cta: "Scan MRI",
    icon: "brain",
  },
  {
    badge: "NLP Pipeline",
    badgeMod: "nlp",
    name: "Clinical Report Analyzer",
    desc: "Biomedical NER pipeline powered by HuggingFace. Extracts diseases, symptoms, medications, and findings from medical reports.",
    tags: ["HuggingFace", "BioBERT", "Entity Extraction"],
    cta: "Analyze Report",
    icon: "document",
  },
];

const CANCERS = [
  { type: "Brain Cancer",    color: "purple", icon: "🧠", classes: ["Glioma", "Meningioma", "Pituitary", "No Tumor"] },
  { type: "Leukemia (ALL)",  color: "red",    icon: "🩸", classes: ["Benign", "Early", "Pre", "Pro"] },
  { type: "Breast Cancer",   color: "pink",   icon: "🎀", classes: ["Benign", "Malignant", "Normal"] },
  { type: "Cervical Cancer", color: "amber",  icon: "🔬", classes: ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial"] },
  { type: "Kidney Cancer",   color: "blue",   icon: "💧", classes: ["Normal", "Tumor"] },
  { type: "Lung & Colon",    color: "green",  icon: "🫁", classes: ["Lung Benign", "Lung Adeno", "Lung Squamous", "Colon Adeno", "Colon Benign"] },
  { type: "Lymphoma",        color: "orange", icon: "🧬", classes: ["CLL", "FL", "MCL"] },
  { type: "Oral Cancer",     color: "cyan",   icon: "👄", classes: ["Normal", "OSCC"] },
];

const BOTTOM_FEATURES = [
  { icon: "⚡", title: "Real-time Inference",     desc: "Predictions in under 2 seconds on CPU — no special hardware needed." },
  { icon: "📊", title: "Confidence Scoring",       desc: "Every result shows a confidence percentage with visual progress bars." },
  { icon: "🔒", title: "Privacy by Design",        desc: "Images processed in-memory only. Nothing stored after your session." },
  { icon: "🌐", title: "REST API",                desc: "Clean FastAPI backend with async endpoints for easy integration." },
];

/* SVG icon helper */
function Icon({ name }) {
  const icons = {
    upload: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>,
    brain: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z"/><line x1="9" y1="22" x2="15" y2="22"/><line x1="10" y1="19" x2="14" y2="19"/></svg>,
    chart: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>,
    microscope: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="9" r="3"/><path d="M12 6V2"/><path d="M8.5 8.5 6 11"/><path d="M15.5 8.5 18 11"/><path d="M12 12v4"/><path d="M7 20h10"/><path d="M9 16h6"/></svg>,
    document: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>,
    arrow: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>,
  };
  return <span className="svg-icon">{icons[name]}</span>;
}

export default function Dashboard() {
  return (
    <>
      {/* ════════ HERO ════════ */}
      <section className="hero">
        <div className="hero-glow hero-glow--1" />
        <div className="hero-glow hero-glow--2" />
        <div className="hero-glow hero-glow--3" />
        <div className="hero-content">
          <span className="hero-badge">
            <span className="hero-badge-dot" />
            AI-Powered Medical Imaging
          </span>
          <h1 className="hero-title">
            Intelligent Cancer<br />
            <span className="grad">Detection Platform</span>
          </h1>
          <p className="hero-sub">
            Harness deep learning to classify cancer from medical images and clinical
            reports — covering <strong>8 cancer types</strong>, <strong>26 subclasses</strong>, and
            biomedical NLP entity extraction.
          </p>

          <div className="hero-stat-row">
            <StatCounter value={135000} suffix="+" label="Training Samples" />
            <div className="hero-stat-sep" />
            <StatCounter value={8} label="Cancer Types" />
            <div className="hero-stat-sep" />
            <StatCounter value={26} label="Subclasses" />
            <div className="hero-stat-sep" />
            <StatCounter value={3} label="AI Pipelines" />
          </div>

          <div className="hero-cta-row">
            <Link to="/analyze" className="btn-primary btn-lg">
              <Icon name="upload" /> Start Analysis
            </Link>
            <Link to="/about" className="btn-secondary btn-lg">Learn More</Link>
          </div>
        </div>
      </section>

      {/* ════════ HOW IT WORKS ════════ */}
      <section className="section section--steps">
        <div className="section-container">
          <div className="section-header">
            <span className="section-eyebrow">Workflow</span>
            <h2>How It Works</h2>
            <p>Three simple steps from upload to diagnosis insight</p>
          </div>
          <div className="steps-row">
            {STEPS.map((s, i) => (
              <div key={s.num} className="step-card">
                <div className="step-num">{s.num}</div>
                <div className="step-icon-wrap">
                  <Icon name={s.icon} />
                </div>
                <h3>{s.title}</h3>
                <p>{s.desc}</p>
                {i < STEPS.length - 1 && <div className="step-connector" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ════════ MODELS ════════ */}
      <section className="section section--models">
        <div className="section-container">
          <div className="section-header">
            <span className="section-eyebrow">AI Models</span>
            <h2>Three Powerful Pipelines</h2>
            <p>Purpose-built deep learning models for multi-modal cancer analysis</p>
          </div>
          <div className="models-grid">
            {MODELS.map((m) => (
              <div key={m.name} className="model-card">
                <div className="model-card-top">
                  <span className={`model-badge model-badge--${m.badgeMod}`}>{m.badge}</span>
                  <span className="model-status">
                    <span className="model-status-dot" /> Active
                  </span>
                </div>
                <div className="model-icon-wrap">
                  <Icon name={m.icon} />
                </div>
                <h3 className="model-name">{m.name}</h3>
                <p className="model-desc">{m.desc}</p>
                <div className="model-tags">
                  {m.tags.map(t => <span key={t} className="model-tag">{t}</span>)}
                </div>
                <Link to="/analyze" className="model-cta">
                  {m.cta} <Icon name="arrow" />
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ════════ CANCER TYPES ════════ */}
      <section className="section section--cancers">
        <div className="section-container">
          <div className="section-header">
            <span className="section-eyebrow">Coverage</span>
            <h2>Supported Cancer Types</h2>
            <p>Our models detect and classify the following malignancies and conditions</p>
          </div>
          <div className="cancer-grid">
            {CANCERS.map(({ type, color, icon, classes }) => (
              <div key={type} className={`cancer-card cancer-card--${color}`}>
                <div className="cancer-card-head">
                  <span className="cancer-card-icon">{icon}</span>
                  <h3>{type}</h3>
                </div>
                <div className="cancer-card-pills">
                  {classes.map(c => (
                    <span key={c} className={`cancer-pill cancer-pill--${color}`}>{c}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ════════ FEATURES BAR ════════ */}
      <section className="section section--features-bar">
        <div className="section-container">
          <div className="features-bar-grid">
            {BOTTOM_FEATURES.map(({ icon, title, desc }) => (
              <div key={title} className="feature-bar-item">
                <span className="feature-bar-icon">{icon}</span>
                <div>
                  <h4>{title}</h4>
                  <p>{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ════════ CTA BANNER ════════ */}
      <section className="section section--cta-banner">
        <div className="section-container">
          <div className="cta-banner">
            <div className="cta-banner-glow" />
            <h2>Ready to Analyze?</h2>
            <p>Upload a medical scan or clinical report and get instant AI-powered results.</p>
            <Link to="/analyze" className="btn-primary btn-lg">
              <Icon name="upload" /> Start Free Analysis
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
