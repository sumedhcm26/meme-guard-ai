"""
utils/config.py
─────────────────────────────────────────────────────────────
Central configuration for the Meme Hate Detection system.
All hyper-parameters and paths are defined here so that
every other module imports from a single source of truth.

Kaggle output files supported
──────────────────────────────
Place any of these inside the  model/  folder:

  OPTION A — Ensemble (RECOMMENDED, highest accuracy)
    model/ensemble_models.pt        ← list of 3 state-dicts
    (also copy ensemble_meta.json → model/ensemble_meta.json  [optional])

  OPTION B — Single best seed
    model/best_model_seed42.pt      ← rename to model/model.pth
        OR
    model/model.pth                 ← already renamed

  OPTION C — Final model
    model/final_model.pt            ← rename to model/model.pth

The predictor auto-detects which file is present in this priority order:
  ensemble_models.pt  >  model.pth  >  best_model_seed42.pt  >  final_model.pt
"""

import json
import os

# ──────────────────────────────────────────────────────────────
# CLIP Backbone
# ──────────────────────────────────────────────────────────────
CLIP_MODEL    = "ViT-L-14"      # ViT-L/14 — strongest public CLIP variant
CLIP_PRETRAIN = "openai"        # OpenAI weights
EMBED_DIM     = 768             # ViT-L/14 output dimension

# ──────────────────────────────────────────────────────────────
# Model Architecture  (must match training exactly)
# ──────────────────────────────────────────────────────────────
MODEL_CFG = dict(
    embed_dim    = EMBED_DIM,
    fusion_heads = 8,
    hidden_dim   = 512,
    dropout      = 0.4,
)

# ──────────────────────────────────────────────────────────────
# File Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(BASE_DIR, "model")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

# All recognised model file locations (checked in priority order)
ENSEMBLE_MODEL_PATH   = os.path.join(MODEL_DIR, "ensemble_models.pt")
SINGLE_MODEL_PATH     = os.path.join(MODEL_DIR, "model.pth")
SEED42_MODEL_PATH     = os.path.join(MODEL_DIR, "best_model_seed42.pt")
SEED55_MODEL_PATH     = os.path.join(MODEL_DIR, "best_model_seed55.pt")
SEED68_MODEL_PATH     = os.path.join(MODEL_DIR, "best_model_seed68.pt")
FINAL_MODEL_PATH      = os.path.join(MODEL_DIR, "final_model.pt")
ENSEMBLE_META_PATH    = os.path.join(MODEL_DIR, "ensemble_meta.json")

# Legacy alias kept for backwards compatibility
MODEL_PATH = SINGLE_MODEL_PATH


def _load_threshold_from_meta() -> float:
    """
    Read the optimal threshold saved by the training notebook
    (ensemble_meta.json → best_threshold).
    Falls back to 0.50 if the file is absent.
    """
    if os.path.exists(ENSEMBLE_META_PATH):
        try:
            with open(ENSEMBLE_META_PATH) as f:
                meta = json.load(f)
            thresh = float(meta.get("best_threshold", 0.50))
            return thresh
        except Exception:
            pass
    return 0.50


# ──────────────────────────────────────────────────────────────
# Inference Settings
# ──────────────────────────────────────────────────────────────
# Uses the optimal threshold from ensemble_meta.json if available,
# otherwise falls back to 0.50.
PREDICTION_THRESHOLD = _load_threshold_from_meta()

MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 MB max upload

# ──────────────────────────────────────────────────────────────
# Allowed Image Formats
# ──────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}

# ──────────────────────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Not Hateful", 1: "Hateful"}
