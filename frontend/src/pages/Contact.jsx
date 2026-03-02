import { useState } from "react";

const INFO_CARDS = [
  {
    icon: (
      <svg className="svg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      </svg>
    ),
    title: "General Inquiries",
    text: "Questions about the project, its architecture, or research methodology.",
  },
  {
    icon: (
      <svg className="svg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
    ),
    title: "Bug Reports",
    text: "Found something that doesn't work? Let us know — include steps to reproduce.",
  },
  {
    icon: (
      <svg className="svg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    ),
    title: "Collaboration",
    text: "Interested in extending this work? We welcome academic collaborations.",
  },
];

export default function Contact() {
  const [sent, setSent] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setSent(true);
    setTimeout(() => setSent(false), 3000);
  };

  return (
    <section className="contact-section">
      <div className="contact-inner">
        {/* ── Header ── */}
        <div className="contact-heading">
          <span className="contact-eyebrow">Get in Touch</span>
          <h2 className="contact-title">We'd Love to <span className="grad">Hear From You</span></h2>
          <p className="contact-lead">
            Have a question, found a bug, or want to collaborate on this
            research? Send us a message and we'll reply as soon as possible.
          </p>
        </div>

        {/* ── Info Cards ── */}
        <div className="contact-info-row">
          {INFO_CARDS.map((c) => (
            <div key={c.title} className="contact-info-card">
              <div className="contact-info-icon">{c.icon}</div>
              <h4>{c.title}</h4>
              <p>{c.text}</p>
            </div>
          ))}
        </div>

        {/* ── Form ── */}
        <form className="contact-form" onSubmit={handleSubmit}>
          <h3 className="contact-form-title">Send a Message</h3>
          <div className="contact-form-row">
            <div className="contact-field">
              <label>Name</label>
              <input type="text" placeholder="Your full name" required />
            </div>
            <div className="contact-field">
              <label>Email</label>
              <input type="email" placeholder="your@email.com" required />
            </div>
          </div>
          <div className="contact-field">
            <label>Subject</label>
            <input type="text" placeholder="e.g. Bug report, Collaboration, Question" required />
          </div>
          <div className="contact-field">
            <label>Message</label>
            <textarea rows={6} placeholder="Write your message here…" required />
          </div>
          <button
            type="submit"
            className={`btn-primary contact-btn ${sent ? "contact-btn--sent" : ""}`}
          >
            {sent ? "✓  Message Sent" : "Send Message"}
          </button>
        </form>
      </div>
    </section>
  );
}