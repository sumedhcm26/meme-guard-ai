"""
model/predict.py
─────────────────────────────────────────────────────────────
End-to-end inference pipeline for the Hateful Meme Detector.

Supports three loading modes (auto-detected):
  1. ENSEMBLE  — ensemble_models.pt  (list of 3 state-dicts)
                 Averages sigmoid probabilities across all seeds.
                 Best accuracy. ← RECOMMENDED from Kaggle output.

  2. SINGLE    — model.pth  /  best_model_seed42.pt  /  final_model.pt
                 Standard single-model inference.

Auto-detection priority:
  ensemble_models.pt > model.pth > best_model_seed42.pt >
  best_model_seed55.pt > best_model_seed68.pt > final_model.pt

Copy your Kaggle output files into the  model/  directory and the
predictor will pick the best available option automatically.
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import open_clip

from model.model_architecture import build_model
from utils.config import (
    CLIP_MODEL,
    CLIP_PRETRAIN,
    MODEL_CFG,
    PREDICTION_THRESHOLD,
    LABEL_MAP,
    # All recognised weight file paths
    ENSEMBLE_MODEL_PATH,
    SINGLE_MODEL_PATH,
    SEED42_MODEL_PATH,
    SEED55_MODEL_PATH,
    SEED68_MODEL_PATH,
    FINAL_MODEL_PATH,
)
from utils.preprocess import preprocess_image, preprocess_text, extract_features

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _auto_detect_weights() -> tuple[str, bool]:
    """
    Scan the model/ directory and return (path, is_ensemble).

    Priority:
      ensemble_models.pt  →  is_ensemble = True
      model.pth           →  is_ensemble = False
      best_model_seed42.pt→  is_ensemble = False
      best_model_seed55.pt→  is_ensemble = False
      best_model_seed68.pt→  is_ensemble = False
      final_model.pt      →  is_ensemble = False
    """
    candidates = [
        (ENSEMBLE_MODEL_PATH, True),
        (SINGLE_MODEL_PATH,   False),
        (SEED42_MODEL_PATH,   False),
        (SEED55_MODEL_PATH,   False),
        (SEED68_MODEL_PATH,   False),
        (FINAL_MODEL_PATH,    False),
    ]
    for path, is_ens in candidates:
        if os.path.exists(path):
            return path, is_ens
    return "", False


def _load_single_model(path: str, device: torch.device):
    """Load one HatefulMemeModel from a state-dict file."""
    model = build_model(MODEL_CFG).to(device)
    state = torch.load(path, map_location=device)
    # Handle the case where the file is itself a list (e.g. picked wrong file)
    if isinstance(state, list):
        state = state[0]
    model.load_state_dict(state)
    model.eval()
    logger.info("  loaded single model from %s", os.path.basename(path))
    return model


def _load_ensemble_models(path: str, device: torch.device) -> list:
    """
    Load all models from ensemble_models.pt.

    The training notebook saves:
        torch.save([m.state_dict() for m in trained_models], path)

    So the file is a Python list of state-dicts.
    """
    payload = torch.load(path, map_location=device)

    if not isinstance(payload, list):
        # Fallback: single state-dict saved in the ensemble slot
        logger.warning("ensemble_models.pt contained a single state-dict; "
                       "treating as single model.")
        m = build_model(MODEL_CFG).to(device)
        m.load_state_dict(payload)
        m.eval()
        return [m]

    models = []
    for i, state_dict in enumerate(payload):
        m = build_model(MODEL_CFG).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)
        logger.info("  loaded ensemble member %d/%d", i + 1, len(payload))

    return models


# ──────────────────────────────────────────────────────────────
# Inference Engine  (singleton — loaded once at startup)
# ──────────────────────────────────────────────────────────────

class MemeHatePredictor:
    """
    Singleton that holds the CLIP backbone + one or more trained
    HatefulMemeModel heads, and exposes a single predict() method.

    Supports both single-model and ensemble inference transparently.
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Device: %s", self.device)

        # ── 1. Load CLIP backbone (frozen) ──────────────────────
        logger.info("Loading CLIP %s (%s) …", CLIP_MODEL, CLIP_PRETRAIN)
        (
            self.clip_model,
            _,
            self.clip_preprocess,
        ) = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAIN
        )
        self.clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.clip_model     = self.clip_model.to(self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        logger.info("CLIP backbone loaded and frozen.")

        # ── 2. Auto-detect & load head model(s) ─────────────────
        weights_path, is_ensemble = _auto_detect_weights()

        if not weights_path:
            logger.warning(
                "No model weights found in model/ directory.\n"
                "Expected one of:\n"
                "  model/ensemble_models.pt   ← best (copy from Kaggle output)\n"
                "  model/model.pth            ← rename best_model_seed42.pt\n"
                "  model/best_model_seed42.pt ← direct copy\n"
                "Running with RANDOM weights — predictions will be meaningless."
            )
            self.models      = [build_model(MODEL_CFG).to(self.device)]
            self.is_ensemble = False
            self.mode        = "random"
        elif is_ensemble:
            logger.info("Loading ENSEMBLE from %s …", weights_path)
            self.models      = _load_ensemble_models(weights_path, self.device)
            self.is_ensemble = True
            self.mode        = f"ensemble ({len(self.models)} models)"
        else:
            logger.info("Loading SINGLE model from %s …", weights_path)
            self.models      = [_load_single_model(weights_path, self.device)]
            self.is_ensemble = False
            self.mode        = f"single ({os.path.basename(weights_path)})"

        self.threshold = PREDICTION_THRESHOLD
        logger.info(
            "MemeHatePredictor ready | mode=%s | threshold=%.2f",
            self.mode, self.threshold,
        )

    # ── Public API ─────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, image_input, text: str) -> dict:
        """
        Run multimodal inference on one meme.

        Args:
            image_input : file path (str / Path) OR PIL.Image
            text        : meme caption string

        Returns dict with keys:
            label, label_id, confidence, probability,
            hateful_prob, not_hateful_prob, mode
        """
        try:
            # ── Preprocessing ───────────────────────────────────
            image_tensor = preprocess_image(
                image_input, self.clip_preprocess, self.device
            )
            text_tokens = preprocess_text(
                text, self.clip_tokenizer, self.device
            )

            # ── CLIP feature extraction ──────────────────────────
            img_feat, txt_feat = extract_features(
                image_tensor, text_tokens, self.clip_model
            )

            # ── Forward pass (single or ensemble) ───────────────
            all_probs = []
            for m in self.models:
                logit, _, _ = m(img_feat, txt_feat)
                all_probs.append(torch.sigmoid(logit).item())

            # Probability averaging (better than logit averaging)
            prob = float(np.mean(all_probs))

            # ── Decision ────────────────────────────────────────
            label_id   = int(prob >= self.threshold)
            label      = LABEL_MAP[label_id]
            confidence = round(
                prob * 100 if label_id == 1 else (1 - prob) * 100, 2
            )

            result = {
                "label"           : label,
                "label_id"        : label_id,
                "confidence"      : confidence,
                "probability"     : round(prob, 6),
                "hateful_prob"    : round(prob * 100, 2),
                "not_hateful_prob": round((1 - prob) * 100, 2),
                "mode"            : self.mode,
            }

            logger.info(
                "Prediction: %s | prob=%.4f | conf=%.2f%% | mode=%s",
                label, prob, confidence, self.mode,
            )
            return result

        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            raise


# ──────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────
_predictor: MemeHatePredictor | None = None


def get_predictor() -> MemeHatePredictor:
    """Return (and lazily create) the module-level singleton."""
    global _predictor
    if _predictor is None:
        logger.info("Initialising MemeHatePredictor …")
        _predictor = MemeHatePredictor()
    return _predictor
