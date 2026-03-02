import { Link } from "react-router-dom";

const TECH = [
  { name: "PyTorch",      desc: "Deep learning framework for model training & inference", color: "amber" },
  { name: "MobileNetV2",  desc: "Efficient CNN backbone pretrained on ImageNet",          color: "purple" },
  { name: "HuggingFace",  desc: "Biomedical NER pipeline (BioBERT/ScispaCy)",             color: "cyan" },
  { name: "FastAPI",      desc: "Async REST backend with automatic docs",                 color: "green" },
  { name: "React + Vite", desc: "Modern frontend with hot-module reloading",              color: "blue" },
  { name: "Python 3.11",  desc: "Backend runtime with type-safe codebase",                color: "amber" },
];

const TIMELINE = [
  { phase: "Research",    desc: "Dataset curation, literature review, model architecture selection" },
  { phase: "Training",    desc: "135K+ image training with data augmentation, hyperparameter tuning" },
  { phase: "Integration", desc: "REST API development, NLP pipeline, frontend build" },
  { phase: "Testing",     desc: "Cross-validation, edge case testing, performance profiling" },
];

export default function About() {
  return (
    <section className="about-section">
      <div className="about-inner">

        {/* ── Header ── */}
        <div className="about-heading">
          <span className="about-eyebrow">About the Project</span>
          <h2 className="about-title">Built for <span className="grad">Cancer Research</span></h2>
          <p className="about-lead">
            MediScan AI is a final-year research project that combines deep learning
            and biomedical NLP to classify cancer from medical images and clinical
            reports — designed for education, exploration, and research.
          </p>
        </div>

        {/* ── Stats ── */}
        <div className="about-stats-row">
          <div className="about-stat">
            <span className="about-stat-value">135K+</span>
            <span className="about-stat-label">Training Images</span>
          </div>
          <div className="about-stat-divider" />
          <div className="about-stat">
            <span className="about-stat-value">8</span>
            <span className="about-stat-label">Cancer Types</span>
          </div>
          <div className="about-stat-divider" />
          <div className="about-stat">
            <span className="about-stat-value">26</span>
            <span className="about-stat-label">Subclasses</span>
          </div>
          <div className="about-stat-divider" />
          <div className="about-stat">
            <span className="about-stat-value">3</span>
            <span className="about-stat-label">AI Pipelines</span>
          </div>
        </div>

        {/* ── Main Content — 2-col ── */}
        <div className="about-two-col">
          <div className="about-card about-card--wide">
            <div className="about-card-icon">🎯</div>
            <h3>What We Built</h3>
            <p>
              Two computer-vision models trained on histopathology and brain MRI
              images, plus a biomedical NLP pipeline — all served via a fast REST
              API and a clean React interface. The platform can classify
              across <strong>8 cancer types</strong> and <strong>26 subclasses</strong> in
              under 2 seconds.
            </p>
            <div className="about-card-divider" />
            <h4 className="about-card-subtitle">Development Timeline</h4>
            <div className="about-timeline">
              {TIMELINE.map((t, i) => (
                <div key={t.phase} className="about-timeline-item">
                  <div className="about-timeline-marker">{i + 1}</div>
                  <div>
                    <strong>{t.phase}</strong>
                    <p>{t.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="about-side-stack">
            <div className="about-card">
              <div className="about-card-icon">🧠</div>
              <h3>How It Works</h3>
              <p>
                Upload a medical image and the classifier returns a cancer type,
                subclass, and confidence score. Upload a clinical report and the NLP
                pipeline highlights diseases, symptoms, medications, and findings.
              </p>
              <ul className="about-list">
                <li><span className="about-pill about-pill--purple">Glioma</span> Brain MRI — malignant</li>
                <li><span className="about-pill about-pill--blue">Meningioma</span> Brain MRI — typically benign</li>
                <li><span className="about-pill about-pill--amber">Pituitary</span> Brain MRI — pituitary gland</li>
                <li><span className="about-pill about-pill--green">No Tumor</span> Brain MRI — healthy scan</li>
              </ul>
            </div>

            <div className="about-card about-card--disclaimer">
              <div className="about-card-icon">⚠️</div>
              <h3>Research Use Only</h3>
              <p>
                MediScan AI is an <strong>academic prototype</strong> — not a certified
                medical device. It has <strong>not been validated clinically</strong> and
                must not guide medical decisions. Always consult a qualified healthcare
                professional.
              </p>
            </div>
          </div>
        </div>

        {/* ── Tech Stack ── */}
        <div className="about-tech-section">
          <h3 className="about-section-title">Technology Stack</h3>
          <div className="about-tech-cards">
            {TECH.map(t => (
              <div key={t.name} className={`about-tech-card about-tech-card--${t.color}`}>
                <span className="about-tech-name">{t.name}</span>
                <span className="about-tech-desc">{t.desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── CTA ── */}
        <div className="about-footer-cta">
          <p>Want to see it in action?</p>
          <Link to="/analyze" className="btn-primary">Try the Analyzer</Link>
        </div>
      </div>
    </section>
  );
}
