"""
=============================================================
  Trustify v2.0 — FastAPI Backend
  Authors: Chandan Upadhyay | Gaurav Maurya | Harshvardhan Vaishnav
  JECRC University, Jaipur | 2025-26

  Endpoints:
    GET  /                  → Frontend HTML
    GET  /api/health        → Server + model status
    POST /api/predict       → Fast ML-only prediction
    POST /api/verify        → Full verification (ML + Internet + LLM)
    POST /api/explain       → Detailed explanation with reasoning
    GET  /api/history       → Recent analyses
    DELETE /api/history     → Clear history
=============================================================
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Add backend dir to path ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from services.nlp_service import ml_predictor, full_nlp_analysis
from services.search_service import verify_claim_online
from services.llm_service import analyze_with_gemini, heuristic_llm_fallback
from services.scoring_service import compute_final_score
from services.fact_checker import check_facts

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Trustify API",
    description="AI-Powered Fake News Detection System",
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static files ─────────────────────────────────────────────────────
FRONTEND_DIR = config.FRONTEND_DIR
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── In-memory history ──────────────────────────────────────────────────────
analysis_history = []

# ── Request/Response Models ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000,
                      description="News headline or article text to analyze")
    verify_online: bool = Field(False, description="Enable internet verification")
    use_llm:       bool = Field(False, description="Enable LLM analysis")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    authors: list
    university: str
    model_accuracy: Optional[float]
    algorithms: list
    api_keys_configured: dict


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML file."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
        <html><body style='font-family:monospace;padding:40px;background:#0d1117;color:#e6edf3'>
        <h1>🛡️ Trustify API Running</h1>
        <p>Frontend not found. Place index.html in /frontend folder.</p>
        <p><a href='/docs' style='color:#00d4ff'>→ View API Docs</a></p>
        </body></html>
    """)


@app.get("/api/health")
async def health():
    """Health check — returns server and model status."""
    meta = ml_predictor.metadata
    return {
        "status":               "ok",
        "model_loaded":         ml_predictor.model is not None,
        "version":              config.VERSION,
        "authors":              config.AUTHORS,
        "university":           config.UNIVERSITY,
        "model_type":           meta.get("model_type", "Not loaded"),
        "model_accuracy":       meta.get("accuracy"),
        "algorithms":           meta.get("algorithms", []),
        "training_samples":     meta.get("training_samples", 0),
        "api_keys_configured": {
            "newsapi":  bool(config.NEWS_API_KEY),
            "serpapi":  bool(config.SERP_API_KEY),
            "gemini":   bool(config.GEMINI_API_KEY),
        },
        "features": {
            "internet_verification": True,
            "llm_analysis":          bool(config.GEMINI_API_KEY),
            "hindi_support":         True,
            "ensemble_model":        True,
        }
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    Fast prediction endpoint — ML model only.
    Response: < 200ms
    Use for: real-time input feedback
    """
    text = req.text.strip()
    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Text too short")

    start = time.time()

    # Run NLP analysis (always fast, no network)
    nlp  = full_nlp_analysis(text)
    ml   = ml_predictor.predict(text)
    fact = check_facts(text)

    internet_result = {"available": False, "fake_probability": 50.0, "real_probability": 50.0,
                       "sources_count": 0, "supporting": 0, "contradicting": 0, "sources": [],
                       "source_type": "none"}

    llm   = heuristic_llm_fallback(text, ml["fake_probability"], nlp["nlp_fake_probability"])
    final = compute_final_score(ml, internet_result, llm, nlp, fact)
    elapsed = round((time.time() - start) * 1000, 1)

    result = {
        **final,
        "mode":           "fast",
        "processing_ms":  elapsed,
        "text_preview":   text[:120] + ("..." if len(text) > 120 else ""),
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    _save_history(result, text)
    return result


@app.post("/api/verify")
async def verify(req: PredictRequest):
    """
    Full verification endpoint — ML + Internet + LLM.
    Response: 3-8 seconds (network dependent)
    Use for: thorough fact-checking
    """
    text = req.text.strip()
    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Text too short")

    start = time.time()

    # Run all analyses concurrently
    nlp  = full_nlp_analysis(text)
    ml   = ml_predictor.predict(text)
    fact = check_facts(text)

    # search_service now handles internet search + ML on articles + Gemini internally
    internet_result = await verify_claim_online(text, ml_predictor_ref=ml_predictor)

    # LLM heuristic still used as supplementary signal when Gemini not configured
    llm_result = heuristic_llm_fallback(text, ml["fake_probability"], nlp["nlp_fake_probability"])

    final = compute_final_score(ml, internet_result, llm_result, nlp, fact)
    elapsed = round((time.time() - start) * 1000, 1)

    result = {
        **final,
        "mode":          "full_verification",
        "processing_ms": elapsed,
        "text_preview":  text[:120] + ("..." if len(text) > 120 else ""),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    _save_history(result, text)
    return result


async def _run_llm(text, ml, nlp):
    """Run LLM — Gemini if configured, else heuristic."""
    if config.GEMINI_API_KEY:
        return await analyze_with_gemini(text)
    return heuristic_llm_fallback(text, ml["fake_probability"], nlp["nlp_fake_probability"])


@app.post("/api/explain")
async def explain(req: PredictRequest):
    """
    Detailed explanation endpoint.
    Same as /verify but focuses on reasoning output.
    """
    result = await verify(req)
    return {
        "verdict":      result["verdict"],
        "confidence":   result["confidence"],
        "explanation":  result["explanation"],
        "reasoning":    result["reasoning"],
        "breakdown":    result["breakdown"],
        "sources":      result.get("sources", []),
        "language":     result.get("language", "en"),
        "timestamp":    result.get("timestamp"),
    }


@app.get("/api/history")
async def get_history():
    return {
        "history": list(reversed(analysis_history)),
        "count":   len(analysis_history),
    }


@app.delete("/api/history")
async def clear_history():
    analysis_history.clear()
    return {"message": "History cleared"}


def _save_history(result: dict, text: str):
    """Save result to in-memory history."""
    analysis_history.append({
        "id":        len(analysis_history) + 1,
        "text":      text[:100] + ("..." if len(text) > 100 else ""),
        "verdict":   result.get("verdict"),
        "is_fake":   result.get("is_fake"),
        "confidence": result.get("confidence"),
        "mode":      result.get("mode"),
        "timestamp": result.get("timestamp"),
    })
    if len(analysis_history) > config.MAX_HISTORY:
        analysis_history.pop(0)


# ── Init message ───────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("  🛡️  Trustify v2.0 — AI Fake News Detection System")
    print("  Chandan Upadhyay | Gaurav Maurya | Harshvardhan Vaishnav")
    print("  JECRC University, Jaipur | 2025-26")
    print("="*60)
    print(f"  🌐 Frontend:  http://localhost:{config.PORT}")
    print(f"  📚 API Docs:  http://localhost:{config.PORT}/docs")
    print(f"  🔍 Predict:   POST /api/predict")
    print(f"  🌐 Verify:    POST /api/verify (full + internet)")
    print(f"  🤖 Explain:   POST /api/explain")
    meta = ml_predictor.metadata
    if meta:
        print(f"\n  Model:    {meta.get('model_type')}")
        print(f"  Accuracy: {meta.get('accuracy')}%")
        print(f"  Algos:    {', '.join(meta.get('algorithms',[]))}")
    print("\n  API Keys:")
    print(f"    NewsAPI:  {'✅' if config.NEWS_API_KEY else '❌ (optional)'}")
    print(f"    SerpAPI:  {'✅' if config.SERP_API_KEY else '❌ (optional)'}")
    print(f"    Gemini:   {'✅' if config.GEMINI_API_KEY else '❌ (optional)'}")
    print("="*60 + "\n")


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
    )
