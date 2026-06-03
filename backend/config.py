"""
=============================================================
  Trustify v3.0 — Central Configuration
  Authors: Gaurav Maurya
=============================================================
  Set your API keys here OR as environment variables.
  Keys marked OPTIONAL work without configuration.
=============================================================
"""

import os

class Config:
    # ── Project Info ───────────────────────────────────────────
    PROJECT_NAME    = "Trustify"
    VERSION         = "3.0.0"
    AUTHORS         = ["Gaurav Maurya"]

    # ── Server ─────────────────────────────────────────────────
    HOST            = "0.0.0.0"
    PORT            = 8000
    DEBUG           = True

    # ── API Keys (set here OR as environment variables) ────────
    # NewsAPI — free at newsapi.org (500 req/day free)
    NEWS_API_KEY    = os.getenv("NEWS_API_KEY")

    # SerpAPI — free tier at serpapi.com (100 searches/month free)
    SERP_API_KEY    = os.getenv("SERP_API_KEY")

    # Gemini — free at aistudio.google.com
    GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

    # ── Paths ──────────────────────────────────────────────────
    BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR       = os.path.join(BASE_DIR, "models")
    DATA_DIR        = os.path.join(BASE_DIR, "data")
    FRONTEND_DIR    = os.path.join(BASE_DIR, "frontend")
    LOG_DIR         = os.path.join(BASE_DIR, "logs")

    # ── Model Paths ────────────────────────────────────────────
    ENSEMBLE_MODEL  = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    TFIDF_PATH      = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    META_PATH       = os.path.join(MODEL_DIR, "model_metadata.pkl")

    # ── Confidence Weights (must sum to 1.0) ──────────────────
    # These match the weighted scoring formula in the requirements
    WEIGHT_ML       = 0.40   # Ensemble ML model
    WEIGHT_INTERNET = 0.30   # Internet source verification
    WEIGHT_LLM      = 0.20   # LLM reasoning (Gemini)
    WEIGHT_NLP      = 0.10   # Sensationalism / NLP signals

    # ── Thresholds ─────────────────────────────────────────────
    CONFIDENCE_HIGH     = 80   # High confidence cutoff
    CONFIDENCE_MODERATE = 60   # Moderate confidence cutoff
    UNCERTAINTY_THRESH  = 55   # Below this → "Estimated"

    # ── Internet Search ────────────────────────────────────────
    MAX_SEARCH_RESULTS  = 5    # Number of sources to fetch
    SEARCH_TIMEOUT      = 8    # Seconds per request

    # ── History ────────────────────────────────────────────────
    MAX_HISTORY         = 50

    # ── Supported Languages ────────────────────────────────────
    SUPPORTED_LANGS     = ["en", "hi"]

config = Config()
