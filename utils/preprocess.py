"""
utils/preprocess.py
─────────────────────────────────────────────────────────────
Image and text preprocessing pipeline that **exactly mirrors**
the training notebook (dl-v4.ipynb):

  • Image  : open_clip.create_model_and_transforms() preprocess
             (resize 224, centre-crop, normalise with CLIP stats)
  • Text   : open_clip.get_tokenizer()  →  token ids of length 77
  • Feature: CLIP encode_image / encode_text  →  L2-normalised 768-d

These helpers are stateless functions that operate on
already-loaded CLIP model objects so the model is initialised
once and reused across requests.
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Image Preprocessing
# ──────────────────────────────────────────────────────────────

def preprocess_image(
    image_input,
    clip_preprocess,
    device: torch.device,
) -> torch.Tensor:
    """
    Load and preprocess an image for CLIP.

    Args:
        image_input    : file path (str/Path) OR a PIL.Image object
        clip_preprocess: the transform returned by
                         open_clip.create_model_and_transforms()
        device         : torch device

    Returns:
        Tensor of shape (1, 3, 224, 224) on `device`
    """
    if isinstance(image_input, (str, Path)):
        logger.debug("Loading image from path: %s", image_input)
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(
            f"image_input must be a path or PIL.Image, got {type(image_input)}"
        )

    # Apply CLIP's standard transform (resize, crop, normalise)
    tensor = clip_preprocess(img)           # (3, 224, 224)
    return tensor.unsqueeze(0).to(device)   # (1, 3, 224, 224)


# ──────────────────────────────────────────────────────────────
# Text Preprocessing
# ──────────────────────────────────────────────────────────────

def preprocess_text(
    text: str,
    clip_tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Tokenise meme text for CLIP.

    Args:
        text           : raw meme caption string
        clip_tokenizer : tokeniser returned by open_clip.get_tokenizer()
        device         : torch device

    Returns:
        Tensor of shape (1, 77) on `device`
    """
    if not text or not text.strip():
        text = " "   # CLIP handles empty sequences poorly; use a space

    logger.debug("Tokenising text (len=%d): %.60s…", len(text), text)

    # clip_tokenizer returns (1, 77) for a list of one string
    tokens = clip_tokenizer([text.strip()])   # (1, 77)
    return tokens.to(device)


# ──────────────────────────────────────────────────────────────
# CLIP Feature Extraction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    image_tensor: torch.Tensor,
    text_tokens:  torch.Tensor,
    clip_model,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the frozen CLIP backbone to obtain L2-normalised embeddings.

    Args:
        image_tensor : (1, 3, 224, 224)
        text_tokens  : (1, 77)
        clip_model   : frozen open_clip model

    Returns:
        img_feat : (1, 768)  L2-normalised image embedding
        txt_feat : (1, 768)  L2-normalised text  embedding
    """
    clip_model.eval()

    img_feat = clip_model.encode_image(image_tensor)   # (1, 768)
    txt_feat = clip_model.encode_text(text_tokens)     # (1, 768)

    # L2-normalise — CLIP convention used during training
    img_feat = F.normalize(img_feat.float(), dim=-1)
    txt_feat = F.normalize(txt_feat.float(), dim=-1)

    return img_feat, txt_feat


# ──────────────────────────────────────────────────────────────
# Validation Helper
# ──────────────────────────────────────────────────────────────

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check that the uploaded filename has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )
