"""
app.py
─────────────────────────────────────────────────────────────
Flask backend for the AI-Powered Multimodal Meme Hate Detection System.

Endpoints:
  GET  /           → serve the main UI (templates/index.html)
  POST /predict    → accept image + text, return JSON prediction
  GET  /health     → health check (for deployment)

Run:
  python app.py
  → http://127.0.0.1:5000
"""

import logging
import os
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

# ── Local imports ──────────────────────────────────────────────
from utils.config import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, UPLOAD_FOLDER
from model.predict import get_predictor

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Flask application factory
# ──────────────────────────────────────────────────────────────
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER

    # Ensure upload directory exists
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

    # ── Warm up predictor at startup ──────────────────────────
    logger.info("Warming up the predictor (loading CLIP + model) …")
    predictor = get_predictor()
    logger.info("Predictor ready.")

    # ──────────────────────────────────────────────────────────
    # Routes
    # ──────────────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def index():
        """Serve the main UI."""
        return render_template("index.html")

    # ── /predict  ─────────────────────────────────────────────
    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Accept a multipart/form-data POST with:
          • file  : meme image
          • text  : meme caption

        Returns JSON:
          {
            "success"     : bool,
            "label"       : "Hateful" | "Not Hateful",
            "label_id"    : 1 | 0,
            "confidence"  : float,       # confidence in the predicted class
            "hateful_prob": float,       # P(hateful) * 100
            "not_hateful_prob": float,
            "probability" : float,       # raw sigmoid output
            "image_url"   : str | null,  # URL to preview the uploaded image
            "error"       : str | null
          }
        """
        # ── 1. Validate image ──────────────────────────────────
        if "file" not in request.files:
            return _error_response("No image file provided.", 400)

        file = request.files["file"]
        if file.filename == "":
            return _error_response("No image selected.", 400)

        filename_lower = file.filename.lower()
        ext = filename_lower.rsplit(".", 1)[-1] if "." in filename_lower else ""
        if ext not in ALLOWED_EXTENSIONS:
            return _error_response(
                f"Invalid file type '.{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                415,
            )

        # ── 2. Validate text ───────────────────────────────────
        text = request.form.get("text", "").strip()
        if not text:
            return _error_response("Meme text cannot be empty.", 400)

        # ── 3. Save image with a UUID name ────────────────────
        safe_name  = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path   = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)
        logger.info("Saved upload → %s", save_path)

        # ── 4. Run prediction ─────────────────────────────────
        try:
            result = predictor.predict(save_path, text)
        except Exception as exc:
            logger.exception("Prediction error: %s", exc)
            return _error_response(f"Prediction failed: {exc}", 500)

        # ── 5. Build response ──────────────────────────────────
        image_url = f"/static/uploads/{unique_name}"
        return jsonify(
            {
                "success"         : True,
                "label"           : result["label"],
                "label_id"        : result["label_id"],
                "confidence"      : result["confidence"],
                "hateful_prob"    : result["hateful_prob"],
                "not_hateful_prob": result["not_hateful_prob"],
                "probability"     : result["probability"],
                "image_url"       : image_url,
                "error"           : None,
            }
        )

    # ── /health  ──────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        """Simple health-check endpoint for container deployments."""
        return jsonify({"status": "ok", "model_loaded": True})

    # ── Static upload serving ─────────────────────────────────
    @app.route("/static/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    # ── 413 handler  ──────────────────────────────────────────
    @app.errorhandler(413)
    def too_large(_):
        return _error_response(
            f"File too large. Maximum size is "
            f"{MAX_CONTENT_LENGTH // (1024*1024)} MB.",
            413,
        )

    return app


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────
def _error_response(message: str, status: int):
    return (
        jsonify({"success": False, "error": message}),
        status,
    )


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
