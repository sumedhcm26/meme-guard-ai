# 🛡️ MemeGuard AI — Multimodal Meme Hate Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)](https://pytorch.org)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--L%2F14-purple)](https://openai.com/research/clip)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000?logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **AI-Powered Multimodal Hate Detection** — Analyses meme images *and* text together using OpenAI's CLIP ViT-L/14 backbone with a custom cross-attention fusion head to classify memes as **Hateful** or **Not Hateful**.

---

## 📸 Demo

| Upload a Meme | Analysis Result |
|---|---|
| Drag & drop any meme image | Instant label + confidence score |

---

## 🧠 Model Architecture

```
Input:
  Image ──► CLIP ViT-L/14 (frozen) ──► img_feat (768-d, L2-normalised)
  Text  ──► CLIP TextEncoder (frozen) ──► txt_feat (768-d, L2-normalised)

Head (trained):
  img_feat ──► AdapterBlock ──► adapted_img (768-d)
  txt_feat ──► AdapterBlock ──► adapted_txt (768-d)

  CrossModalFusion:
    adapted_img ──► MultiheadAttention ──► img_ctx  (image-in-text-context)
    adapted_txt ──► MultiheadAttention ──► txt_ctx  (text-in-image-context)
    fused = [img_ctx ‖ txt_ctx ‖ img_ctx ⊙ txt_ctx]  → (B, 3×768)

  Classifier:
    FC(2304 → 512) → GELU → Dropout(0.4)
    FC(512  → 256) → GELU → Dropout(0.3)
    FC(256  → 1)   → sigmoid → P(hateful)
```

### Key Design Choices

| Component | Detail |
|---|---|
| **Backbone** | CLIP ViT-L/14 @ 224px (307M params, completely frozen) |
| **Fusion** | Bidirectional cross-attention + Hadamard product |
| **Loss** | BCE + Label Smoothing (ε=0.1) + Unimodal Penalty (λ=0.1) |
| **Regularisation** | Dropout, Mixup (α=0.2), Feature-space noise augmentation |
| **Optimiser** | AdamW + OneCycleLR (10% warmup → cosine decay) |
| **Dataset** | Facebook Hateful Memes Challenge |

---

## 🗂️ Project Structure

```
meme-hate-detector/
├── app.py                    # Flask application (entry point)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── model/
│   ├── __init__.py
│   ├── model_architecture.py # HatefulMemeModel, AdapterBlock, CrossModalFusion
│   ├── predict.py            # MemeHatePredictor singleton + inference pipeline
│   └── model.pth             # ← PLACE YOUR TRAINED WEIGHTS HERE
│
├── utils/
│   ├── __init__.py
│   ├── config.py             # All constants, paths, hyperparameters
│   └── preprocess.py         # Image & text preprocessing (mirrors training)
│
├── templates/
│   └── index.html            # Jinja2 template for the UI
│
└── static/
    ├── styles.css            # Modern dark-themed CSS
    ├── script.js             # Frontend JS (drag-drop, API calls, animations)
    └── uploads/              # Temporary image storage (auto-created)
```

---

## ⚡ Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/yourusername/meme-hate-detector.git
cd meme-hate-detector
```

### 2. Create & activate virtual environment

```bash
# Python 3.10+
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
# CPU-only PyTorch (recommended for most machines)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# --- OR ---
# CUDA 12.1 (if you have an NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Place model weights

```bash
# Copy your trained weights file to:
cp /path/to/best_model_seed42.pt model/model.pth
```

> **Note:** If `model/model.pth` is absent the server still starts but predictions will be random (random-weight initialisation). A warning is logged.

### 5. Run the server

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🔌 REST API

### `POST /predict`

Accepts `multipart/form-data` with:

| Field  | Type   | Required | Description                      |
|--------|--------|----------|----------------------------------|
| `file` | binary | ✅       | Meme image (PNG/JPG/JPEG/GIF/WEBP/BMP, ≤ 16 MB) |
| `text` | string | ✅       | Meme caption / overlay text      |

**Response (200 OK)**:

```json
{
  "success":          true,
  "label":            "Hateful",
  "label_id":         1,
  "confidence":       87.34,
  "hateful_prob":     87.34,
  "not_hateful_prob": 12.66,
  "probability":      0.873421,
  "image_url":        "/static/uploads/abc123_meme.jpg",
  "error":            null
}
```

**Error response**:

```json
{
  "success": false,
  "error":   "No image file provided."
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## 🧪 cURL Example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@/path/to/meme.jpg" \
  -F "text=Your meme caption here"
```

---

## 📦 Python Client Example

```python
import requests

resp = requests.post(
    "http://127.0.0.1:5000/predict",
    files={"file": open("meme.jpg", "rb")},
    data={"text": "The meme caption goes here"},
)
result = resp.json()
print(f"{result['label']}  ({result['confidence']:.1f}% confidence)")
```

---

## 🚀 Deployment

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
docker build -t meme-hate-detector .
docker run -p 5000:5000 -v $(pwd)/model:/app/model meme-hate-detector
```

### Production (Gunicorn)

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 "app:create_app()"
```

> Use `-w 1` (single worker) to avoid loading CLIP into memory multiple times.

---

## 📊 Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 40 (early stopping, patience=7) |
| Batch size | 32 |
| Learning rate | 3×10⁻⁴ (OneCycleLR) |
| Weight decay | 1×10⁻² |
| Label smoothing | 0.10 |
| Mixup α | 0.20 |
| Gradient clip | 1.0 |
| Dropout | 0.40 (main), 0.30 (second layer) |
| Unimodal λ | 0.10 |

---

## ⚠️ Disclaimer

This system is intended for **research and educational purposes only**. Automated content moderation tools should be used as an aid to — not a replacement for — human review. The model may make errors and should not be deployed in production safety-critical systems without further validation.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [OpenAI CLIP](https://openai.com/research/clip) for the ViT-L/14 backbone
- [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) for the open-source CLIP implementation
- [Facebook Hateful Memes Challenge](https://hatefulmemeschallenge.com/) for the dataset
