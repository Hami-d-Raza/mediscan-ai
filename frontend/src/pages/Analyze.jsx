import { useState } from "react";
import ImageUpload from "../components/ImageUpload";
import ReportUpload from "../components/ReportUpload";

export default function Analyze() {
  const [activeTab, setActiveTab] = useState("image");

  return (
    <section className="analyze-section">
      <div className="analyze-heading">
        <span className="analyze-eyebrow">AI Analysis</span>
        <h2 className="analyze-title">Start Your <span className="grad">Diagnosis</span></h2>
        <p className="analyze-lead">
          Upload a medical scan for instant cancer classification or a clinical report
          for AI-powered entity extraction.
        </p>
      </div>

      <div className="tab-bar">
        <button
          className={`tab-btn ${activeTab === "image" ? "tab-btn--active" : ""}`}
          onClick={() => setActiveTab("image")}
        >
          <svg className="svg-icon svg-icon--sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
          </svg>
          Medical Scan
        </button>
        <button
          className={`tab-btn ${activeTab === "report" ? "tab-btn--active" : ""}`}
          onClick={() => setActiveTab("report")}
        >
          <svg className="svg-icon svg-icon--sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /><polyline points="10 9 9 9 8 9" />
          </svg>
          Medical Report
        </button>
      </div>

      <div className="panels-grid">
        <div className={`panel-wrap ${activeTab === "image" ? "panel-wrap--active" : ""}`}>
          <ImageUpload />
        </div>
        <div className={`panel-wrap ${activeTab === "report" ? "panel-wrap--active" : ""}`}>
          <ReportUpload />
        </div>
      </div>
    </section>
  );
}
