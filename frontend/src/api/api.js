/**
 * api/api.js — Axios HTTP client for the MediScan AI backend
 *
 * All request paths are relative to VITE_API_BASE_URL (configured in .env).
 *
 * Development flow:
 *   VITE_API_BASE_URL is left empty → axios uses baseURL "/" → the Vite
 *   dev-server proxy (vite.config.js) forwards each request transparently
 *   to http://localhost:8000.  Because the browser only talks to port 3000
 *   there are zero CORS issues during development.
 *
 * Production flow:
 *   Set VITE_API_BASE_URL=https://your-backend.com in .env so axios hits
 *   the real server directly.  Ensure that domain is in ALLOWED_ORIGINS in
 *   backend/app/config.py.
 *
 * IMPORTANT — Do NOT set Content-Type when uploading FormData:
 *   Axios auto-detects "multipart/form-data" and appends the boundary token.
 *   Manually setting the header removes the boundary, causing FastAPI to
 *   reject the request with 422 Unprocessable Entity.
 */
import axios from "axios";

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/";

const api = axios.create({
  baseURL: BASE_URL,
  withCredentials: true,   // mirrors FastAPI allow_credentials=True in CORS
});

/**
 * Check if the backend is reachable.
 * @returns {Promise<{status: string, service: string}>}
 */
export async function healthCheck() {
  const { data } = await api.get("/health");
  return data;
}

/**
 * Analyze a medical image (JPG / PNG).
 * @param {File} file
 * @returns {Promise<{filename, prediction, confidence, class_index, simulated}>}
 */
export async function analyzeImage(file) {
  const form = new FormData();
  form.append("file", file);
  // Do NOT set Content-Type manually — axios infers multipart/form-data
  // and appends the required boundary parameter automatically.
  const { data } = await api.post("/analyze-image", form);
  return data;
}

/**
 * Analyze a Brain MRI image using the dedicated MRI classifier.
 * @param {File} file
 * @returns {Promise<{filename, prediction, cancer_type, confidence, class_index, simulated, description}>}
 */
export async function analyzeBrainMRI(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/analyze-brain-mri", form);
  return data;
}

/**
 * Analyze a medical report (PDF / TXT).
 * @param {File} file
 * @returns {Promise<{filename, char_count, entities, summary, simulated}>}
 */
export async function analyzeReport(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/analyze-report", form);
  return data;
}
