# User & Operator Guide: Bright Future Image Classifier

**Target Audience:** School IT Administrators, DevOps Engineers, and Software Developers.  
**Purpose:** Comprehensive guide for running, maintaining, troubleshooting, and diagnosing the image classification service locally and in production.

---

## 1. Quick Emergency Reference (Cheat Sheet)

If the system is down or you need to inspect an image immediately:

| Objective | Command | Port / Path |
|:---|:---|:---:|
| **Start Local API Server** | `uvicorn src.app:app --host 0.0.0.0 --port 8000` | `http://localhost:8000` |
| **Interactive Swagger Docs** | Open browser to `http://localhost:8000/docs` | `/docs` |
| **Service Health Check** | `curl -f http://localhost:8000/health` | `/health` |
| **Classify an Image (cURL)** | `curl -X POST "http://localhost:8000/predict" -F "file=@photo.jpg"` | `/predict` |
| **Generate Grad-CAM Heatmap** | `python src/gradcam.py --count 5` | `results/gradcam/` |
| **Build & Run Docker Container**| `docker build -t classifier . && docker run -p 8000:8000 classifier` | Port `8000` |
| **Cloud Production URL** | `https://bright-future-image-classifier.onrender.com` | Live on Render |

---

## 2. System Architecture & Prerequisites

### 2.1 Technology Stack
- **Runtime:** Python 3.10+ (Tested on Python 3.11 & 3.14)
- **Frameworks:** FastAPI, Uvicorn, PyTorch, Torchvision, Pydantic, Pillow, Matplotlib
- **Containerization:** Docker (Multi-stage Debian slim, CPU-only PyTorch)
- **Hosting Platform:** Render Web Services (Free Tier, 512 MB RAM)

### 2.2 Directory Layout
```text
bright-future-image-classifier/
├── src/
│   ├── app.py                   # FastAPI REST API (inference service)
│   ├── gradcam.py               # Grad-CAM heatmap visualization engine
│   ├── error_analysis.py        # Automated test error audit script
│   └── train_improved.py        # Model retraining script
├── models/
│   └── improved_resnet18_best.pt# Production ResNet-18 weights (42.7 MB)
├── docs/
│   ├── model_card.md            # Standard ML model card specification
│   ├── user_guide.md            # This operator & developer guide
│   ├── error_analysis_report.md # Error analysis & failure case study
│   └── architecture_decision_log.md # Architectural decisions (ADR)
├── handover/                    # Packaged client delivery bundle
├── Dockerfile                   # Production container definition
├── render.yaml                  # Render cloud infrastructure config
└── requirements.txt             # Pinned Python dependencies
```

---

## 3. Running the FastAPI Service Locally

### Step 1: Set Up Python Environment
```bash
# Clone the repository
git clone https://github.com/Mushtaq0054/bright-future-image-classifier.git
cd bright-future-image-classifier

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Start the Web Server
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
*Expected Terminal Output:*
```text
INFO:     Started server process [1234]
INFO:     Waiting for application startup.
INFO:     Loading fine-tuned model checkpoint from: .../models/improved_resnet18_best.pt
INFO:     Successfully loaded fine-tuned checkpoint weights.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 4. API Endpoints & Request Examples

### 4.1 Health Check (`GET /health`)
Verify that the service is running and the PyTorch model is loaded into memory:
```bash
curl -X GET "http://localhost:8000/health"
```
*Expected JSON Response:*
```json
{
  "status": "healthy",
  "service": "bright-future-image-classifier",
  "model_loaded": true,
  "device": "cpu",
  "target_classes": 10
}
```

### 4.2 Image Prediction (`POST /predict`)
Send an image file as `multipart/form-data`:

#### Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_event_photo.jpg"
```

#### Using Python (`requests` library):
```python
import requests

url = "http://localhost:8000/predict"
file_path = "school_event.jpg"

with open(file_path, "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Result:", response.json())
```

*Expected JSON Response:*
```json
{
  "success": true,
  "predicted_class": "automobile",
  "scene_tag": "playground",
  "confidence": 0.9854,
  "confidence_percentage": "98.54%",
  "top_3_predictions": [
    {"class_name": "automobile", "confidence_percentage": 98.54},
    {"class_name": "truck", "confidence_percentage": 1.21},
    {"class_name": "airplane", "confidence_percentage": 0.15}
  ],
  "inference_time_ms": 14.25,
  "model_version": "ResNet-18-v2-Augmented"
}
```

---

## 5. How to Generate & Interpret Grad-CAM Visualizations

Grad-CAM (Gradient-weighted Class Activation Mapping) produces 2D heatmaps showing which pixels influenced the model's decision. This is critical for auditing suspicious or misclassified photos.

### 5.1 Running Grad-CAM Generation
To generate visual explanations for the top misclassified test samples:
```bash
python src/gradcam.py --count 5
```
Or for custom image paths:
```bash
python src/gradcam.py --csv-path results/misclassified_samples.csv --output-dir results/gradcam/
```

### 5.2 Interpreting the 3-Panel Diagnostic Figure
Saved in `results/gradcam/` (e.g., `gradcam_error_03_cat_as_frog.png`):
1. **Left Panel (Original Image):** The raw photograph input with its ground truth label.
2. **Middle Panel (Grad-CAM Heatmap):** Pure normalized activation map extracted from `model.layer4`. Red/yellow denotes high positive activation; blue/purple denotes zero/negative influence.
3. **Right Panel (Overlay Visualization):** 50% alpha blend of the jet colormap onto the photograph.
   - **Good Model Focus:** Red/yellow activations centered squarely on the subject (e.g. face of animal, chassis of car).
   - **Context/Background Leakage:** Red/yellow activations highlighting surrounding grass, trees, or floor textures rather than the subject.

---

## 6. Containerized Deployment with Docker

### 6.1 Build Local Docker Image
```bash
docker build -t bright-future-image-classifier:latest .
```

### 6.2 Run Docker Container
```bash
docker run -d --name bright-future-app -p 8000:8000 bright-future-image-classifier:latest
```

### 6.3 Verify Running Container
```bash
docker ps
curl http://localhost:8000/health
```

### 6.4 Stop Container
```bash
docker stop bright-future-app
docker rm bright-future-app
```

---

## 7. Cloud Deployment on Render

The service is pre-configured with `render.yaml` for zero-configuration deployment:
- **Live URL:** `https://bright-future-image-classifier.onrender.com`
- **Swagger Documentation:** `https://bright-future-image-classifier.onrender.com/docs`

### Handling Cold Starts on Render Free Tier
Render free-tier instances sleep after 15 minutes of inactivity. When a request arrives after sleep:
- **First Request Delay:** 30 to 50 seconds while the container initializes and PyTorch loads weights into RAM.
- **Subsequent Requests:** 10 to 30 ms response times.
- **Keep-Alive Recommendation:** Use a free uptime monitor (e.g., UptimeRobot) to ping `/health` every 10 minutes if continuous instant availability is required.

---

## 8. Troubleshooting & FAQ (At 3 AM)

| Symptom / Error | Root Cause | Solution |
|:---|:---|:---|
| **`422 Unprocessable Entity` on `/predict`** | Uploaded file is corrupted or not a valid image. | Check that the file is an actual JPG, PNG, or WEBP image. |
| **`400 Bad Request: Unsupported file type`** | Unsupported MIME type (e.g. PDF, MP4, GIF). | Send only `image/jpeg`, `image/png`, or `image/webp`. |
| **`Address already in use: port 8000`** | Another process is occupying port 8000. | Change port: `uvicorn src.app:app --port 8080` or kill the process on 8000. |
| **`ModuleNotFoundError: No module named 'src'`** | Python ran from wrong directory. | Ensure you are in the project root (`bright-future-image-classifier/`). |
| **Container crashes with OOM (Out Of Memory)** | PyTorch downloaded CUDA libraries or allocated excess RAM. | Ensure `Dockerfile` installs CPU-only wheel: `--index-url https://download.pytorch.org/whl/cpu`. |
| **Model weights not found warning in log** | `models/improved_resnet18_best.pt` missing. | Ensure checkpoint is downloaded. App automatically falls back to ImageNet weights. |
