# MediScan AI

An AI-powered medical research platform combining **three distinct AI pipelines** — a 26-class histopathology classifier, a brain MRI tumour classifier, and a biomedical NLP report analyser — delivered through a FastAPI backend and a React + Vite frontend.

> ⚠️ **Research & educational use only.** Not a certified medical device. Do not use for clinical decision-making.

---

## AI Pipelines at a Glance

| Pipeline | Model | Training Data | Classes |
|----------|-------|--------------|---------|
| **Multi-Cancer Classifier** | MobileNetV2 (fine-tuned) | 135K+ histopathology images | 26 subclasses across 8 cancer types |
| **Brain MRI Classifier** | Custom CNN | ~7,000 MRI scans | 4 classes (Glioma, Meningioma, Pituitary, No Tumor) |
| **Clinical NLP Analyser** | HuggingFace BioBERT NER | Pre-trained biomedical model | Diseases · Symptoms · Medications · Findings |

### Supported Cancer Types (Multi-Cancer Model)

| Cancer Type | Subclasses |
|---|---|
| Brain Cancer | `brain_glioma` · `brain_menin` · `brain_tumor` |
| Leukemia (ALL) | `all_benign` · `all_early` · `all_pre` · `all_pro` |
| Breast Cancer | `breast_benign` · `breast_malignant` |
| Cervical Cancer | `cervix_dyk` · `cervix_koc` · `cervix_mep` · `cervix_pab` · `cervix_sfi` |
| Kidney Cancer | `kidney_normal` · `kidney_tumor` |
| Lung & Colon Cancer | `colon_aca` · `colon_bnt` · `lung_aca` · `lung_bnt` · `lung_scc` |
| Lymphoma | `lymph_cll` · `lymph_fl` · `lymph_mcl` |
| Oral Cancer | `oral_normal` · `oral_scc` |

### Training Datasets

| Dataset | Link | Classes | Images |
|---------|------|---------|--------|
| Multi-Cancer | [Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer) | 26 subclasses / 8 types | ~130,000 |
| Brain Tumor MRI | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 4 | ~7,000 |

---

## Model Weights

The `.pt` model files are **not included in the repository** (too large for git). Download them from the GitHub Release and place them in `backend/models/`:

| File | Description | Download |
|------|-------------|----------|
| `mri_classifier.pt` | 4-class Brain MRI CNN | [v1.0.0 Release](https://github.com/Hami-d-Raza/mediscan-ai/releases/download/v1.0.0/mri_classifier.pt) |
| `multi_cancer_classifier.pt` | 26-class MobileNetV2 histopathology model | [v1.0.0 Release](https://github.com/Hami-d-Raza/mediscan-ai/releases/download/v1.0.0/multi_cancer_classifier.pt) |

```bash
# After downloading, place both files here:
backend/models/mri_classifier.pt
backend/models/multi_cancer_classifier.pt
```

---

## Project Structure

```
FYP/
├── .venv/                         # Python virtual environment (git-ignored)
│
├── backend/                       # FastAPI application
│   ├── app/
│   │   ├── main.py                # App entry point, CORS, router registration
│   │   ├── config.py              # Pydantic settings (reads from .env)
│   │   ├── models/                # Pydantic data models
│   │   │   ├── user.py
│   │   │   └── analysis.py
│   │   ├── routes/                # One file per feature area
│   │   │   ├── auth.py            # Register / login (JWT)
│   │   │   ├── health.py          # GET /health
│   │   │   ├── image.py           # POST /analyze-image  (multi-cancer)
│   │   │   ├── mri.py             # POST /analyze-brain-mri
│   │   │   ├── nlp.py             # POST /analyze-report
│   │   │   ├── report.py
│   │   │   └── vision.py
│   │   ├── services/              # Business logic (decoupled from routes)
│   │   │   ├── image_model.py     # MobileNetV2 inference wrapper
│   │   │   ├── mri_model.py       # Brain MRI CNN inference wrapper
│   │   │   ├── nlp_model.py       # HuggingFace BioBERT NER pipeline
│   │   │   ├── nlp_service.py
│   │   │   ├── auth_service.py
│   │   │   └── vision_service.py
│   │   └── utils/                 # Shared helpers
│   │       ├── file_handler.py
│   │       ├── file_utils.py
│   │       └── response_utils.py
│   ├── models/                    # Trained weights & label files
│   │   ├── multi_cancer_classifier.pt   # 26-class MobileNetV2 weights
│   │   ├── mri_classifier.pt            # 4-class Brain MRI CNN weights
│   │   └── class_labels.json            # label index → class name mapping
│   ├── data/                      # Local datasets (git-ignored)
│   │   ├── brain_tumor_mri/
│   │   └── multi_cancer/
│   ├── uploads/                   # Temp upload directory (git-ignored)
│   ├── .env.example               # Environment variable template
│   ├── requirements.txt
│   ├── run.py                     # Convenience script: uvicorn app.main:app
│   └── train_mri.py               # Brain MRI training script
│
├── frontend/                      # React + Vite application
│   ├── src/
│   │   ├── api/
│   │   │   └── api.js             # Axios client — all HTTP calls live here
│   │   ├── components/
│   │   │   ├── Layout.jsx         # Navbar + footer shell (hamburger menu)
│   │   │   ├── ImageUpload.jsx    # Dual-model image upload & results card
│   │   │   └── ReportUpload.jsx   # NLP report upload & entity display card
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Landing page — stats, models, cancer types
│   │   │   ├── Analyze.jsx        # Tabbed analysis page (scan / report)
│   │   │   ├── About.jsx          # Project info, tech stack, disclaimer
│   │   │   └── Contact.jsx        # Contact form + info cards
│   │   ├── App.jsx                # Router configuration
│   │   └── App.css                # All styles (dark glassmorphism theme)
│   ├── .env                       # VITE_API_BASE_URL (empty = use Vite proxy)
│   ├── vite.config.js             # Dev-server proxy → localhost:8000
│   └── package.json
│
├── uploads/                       # Root-level upload fallback (git-ignored)
├── train_colab.ipynb              # Google Colab training notebook
├── test_report.txt                # Sample report for testing NLP pipeline
└── README.md
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10 + | `python --version` |
| Node.js | 18 + | `node --version` |
| npm | 9 + | bundled with Node |

---

## Backend Setup

### 1. Create and activate a virtual environment

```powershell
# From the project root
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux
```

### 2. Install Python dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 3. Configure environment variables (optional)

Copy the provided template and edit as needed:

```powershell
copy backend\.env.example backend\.env
```

Key variables:

```ini
DEBUG=True
SECRET_KEY=your-secret-key-here
UPLOAD_DIR=uploads
CV_MODEL_PATH=models/multi_cancer_classifier.pt
CV_LABELS_PATH=models/class_labels.json
MRI_MODEL_PATH=models/mri_classifier.pt
```

All settings have sensible defaults — the app runs without an `.env` file.

### 4. Place trained model weights

Download the trained weights from your Kaggle notebook output and place them here:

```
backend/models/
  multi_cancer_classifier.pt   ← 26-class MobileNetV2 weights
  mri_classifier.pt            ← 4-class Brain MRI CNN weights
  class_labels.json            ← label index → class name mapping
```

> If a model file is missing the backend starts in **Simulation Mode** and returns deterministic mock predictions (marked ⚠️ Simulated in the UI).

### 5. Start the backend server

```powershell
# From the backend/ directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**.  
Interactive docs: **http://localhost:8000/docs**

---

## Frontend Setup

### 1. Install Node dependencies

```powershell
cd frontend
npm install
```

### 2. Configure the API base URL

The default `.env` file sets `VITE_API_BASE_URL=` (empty).  
This tells axios to use `/` as the base URL, and the Vite dev-server proxy forwards all API requests to `http://localhost:8000` — no CORS configuration needed in development.

```
# frontend/.env
VITE_API_BASE_URL=          # leave empty for development
```

For production, set this to your deployed backend URL:

```
VITE_API_BASE_URL=https://api.mediscan.example.com
```

### 3. Start the frontend dev server

```powershell
npm run dev
```

The app will open at **http://localhost:3000**.

---

## Running Both Servers

Open two terminals:

**Terminal 1 — Backend**
```powershell
cd backend
..\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

**Terminal 2 — Frontend**
```powershell
cd frontend
npm run dev
```

Then open **http://localhost:3000** in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health`            | Health check — confirms the API is running |
| `POST` | `/analyze-image`     | Upload a scan image for multi-cancer CV inference (26 classes) |
| `POST` | `/analyze-brain-mri` | Upload a brain MRI image for 4-class CNN inference |
| `POST` | `/analyze-report`    | Upload a PDF/TXT clinical report for BioBERT NER extraction |
| `POST` | `/auth/register`     | Register a new user account |
| `POST` | `/auth/login`        | Obtain a JWT access token |
| `GET`  | `/docs`              | Swagger UI (interactive API docs) |

### Example — Analyze Image (26-class)

```bash
curl -X POST http://localhost:8000/analyze-image \
  -F "file=@histology_slide.jpg"
```

Response:
```json
{
  "filename":    "histology_slide.jpg",
  "prediction":  "brain_glioma",
  "cancer_type": "Brain Cancer",
  "confidence":  0.91,
  "class_index": 4,
  "simulated":   false
}
```

### Example — Analyze Brain MRI (4-class)

```bash
curl -X POST http://localhost:8000/analyze-brain-mri \
  -F "file=@brain_scan.jpg"
```

Response:
```json
{
  "filename":    "brain_scan.jpg",
  "prediction":  "glioma",
  "cancer_type": "Brain Cancer",
  "confidence":  0.94,
  "class_index": 0,
  "description": "Glioma is a type of malignant brain tumour arising from glial cells.",
  "simulated":   false
}
```

### Example — Analyze Report

```bash
curl -X POST http://localhost:8000/analyze-report \
  -F "file=@report.pdf"
```

Response:
```json
{
  "filename":   "report.pdf",
  "char_count": 2048,
  "entities": {
    "diseases":    ["Diabetes mellitus"],
    "symptoms":    ["Polyuria", "Fatigue"],
    "medications": ["Metformin"],
    "other":       []
  },
  "summary":   "Patient presents with Type 2 Diabetes…",
  "simulated": false
}
```

---

## Supported File Types

| Feature | Accepted formats | Notes |
|---------|-----------------|-------|
| Image analysis (both models) | `.jpg`, `.jpeg`, `.png` | MRI scans, histopathology slides, microscopy images |
| Report analysis | `.pdf`, `.txt` | Clinical notes, pathology and radiology reports |

## Model Architecture

### Multi-Cancer Classifier

| Property | Value |
|----------|-------|
| Base model | MobileNetV2 (ImageNet pretrained) |
| Head | Dropout(0.2) → Linear(1280 → 26) |
| Input size | 224 × 224 RGB |
| Training strategy | Two-phase transfer learning (frozen backbone → full fine-tune) |
| Output | Softmax over 26 subclasses, mapped to 8 cancer types |
| Weights | `backend/models/multi_cancer_classifier.pt` |
| Labels | `backend/models/class_labels.json` |

### Brain MRI Classifier

| Property | Value |
|----------|-------|
| Architecture | Custom CNN |
| Input size | 224 × 224 RGB |
| Output | Softmax over 4 classes (Glioma, Meningioma, Pituitary, No Tumor) |
| Weights | `backend/models/mri_classifier.pt` |

### Clinical NLP Analyser

| Property | Value |
|----------|-------|
| Model | HuggingFace BioBERT-based NER |
| Entities extracted | Diseases, Symptoms, Medications, Findings |
| Input | Plain text extracted from PDF or TXT upload |

---

## Troubleshooting

**"Backend offline" badge in the UI**
- Make sure the FastAPI server is running on port 8000.
- Check that `uvicorn` started without errors in the backend terminal.

**422 Unprocessable Entity on file upload**
- Do not set `Content-Type: multipart/form-data` manually — let axios set it automatically (it must include the `boundary` parameter).

**CORS errors in the browser console**
- In development, use the Vite proxy (keep `VITE_API_BASE_URL` empty).
- In production, add your frontend origin to `ALLOWED_ORIGINS` in `backend/app/config.py`.

**ModuleNotFoundError when starting the backend**
- Make sure the virtual environment is activated: `.venv\Scripts\Activate.ps1`
- Re-run: `pip install -r requirements.txt`

---

## Frontend Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Dashboard | Landing page with animated stats, How It Works, AI models overview, cancer types grid |
| `/analyze` | Analyze | Tabbed interface — Medical Scan tab (dual-model) and Medical Report tab |
| `/about` | About | Project background, development timeline, tech stack cards, disclaimer |
| `/contact` | Contact | Contact form with info cards |

The **Analyze** page includes a **dual-model toggle** in the image upload card: users can switch between the 26-class multi-cancer classifier and the 4-class brain MRI classifier without leaving the page.

---

## Notes

- Results marked **⚠️ Simulated result** mean the model weights file was not found at startup. The backend falls back to deterministic mock predictions. Place the correct `.pt` file in `backend/models/` to enable real inference.
- Both `mri_classifier.pt` (4-class brain MRI) and `multi_cancer_classifier.pt` (26-class histopathology) must be present for all three AI pipelines to operate in live mode.
- See `backend/.env.example` for all configurable environment variables.
- The `train_colab.ipynb` notebook contains the full multi-cancer training pipeline for Google Colab; `backend/train_mri.py` is the local brain MRI training script.
- This project is intended for **research and educational use only**. It is not a certified medical device and must not be used for clinical decision-making.
