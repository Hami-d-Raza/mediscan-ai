const FEATURES = [
  {
    icon: "🔬",
    title: "Multi-Cancer Classifier",
    desc: "Classifies histopathology images across 8 cancer types and 26 subclasses, trained on 135K+ images.",
  },
  {
    icon: "🧠",
    title: "Brain MRI Analysis",
    desc: "Dedicated model for brain MRI scans — detects Glioma, Meningioma, Pituitary tumours, and healthy tissue.",
  },
  {
    icon: "📄",
    title: "Clinical Report NLP",
    desc: "Extracts diseases, symptoms, medications and findings from PDF or text medical reports automatically.",
  },
  {
    icon: "⚡",
    title: "Fast Inference",
    desc: "Predictions delivered in under 2 seconds on CPU — no GPU or special hardware required.",
  },
  {
    icon: "📊",
    title: "Confidence Scores",
    desc: "Every result includes a confidence percentage and visual bar so you can gauge model certainty at a glance.",
  },
  {
    icon: "🔒",
    title: "Privacy First",
    desc: "Uploads are processed in memory only — nothing is stored on the server after your session ends.",
  },
];

export default function Features() {
  return (
    <section className="features-page">
      <div className="features-page-inner">
        <div className="section-header">
          <h2>Features</h2>
          <p>What MediScan AI can do</p>
        </div>
        <div className="features-full-grid">
          {FEATURES.map(({ icon, title, desc }) => (
            <div key={title} className="feature-card feature-card--large">
              <span className="feature-icon">{icon}</span>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
