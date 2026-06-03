"""
LLM Verification Service
Gemini API integration + rich heuristic fallback.
The heuristic is significantly stronger than before so Full Verify
gives meaningfully different results from Fast Check.
"""
import asyncio, aiohttp, json, re
from typing import Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

SYSTEM_PROMPT = """You are Trustify, an expert AI fact-checker. Analyze the news claim and respond ONLY in this exact JSON format (no markdown):
{
  "fake_probability": <0-100>,
  "real_probability": <0-100>,
  "verdict": "FAKE" or "REAL" or "UNCERTAIN",
  "confidence": "High" or "Moderate" or "Low",
  "reasoning": ["reason1", "reason2", "reason3"],
  "explanation": "2-3 sentence explanation"
}"""


async def analyze_with_gemini(text: str) -> Dict:
    if not config.GEMINI_API_KEY:
        return _unavailable()
    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\nClaim: \"{text}\""}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 400},
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{GEMINI_URL}?key={config.GEMINI_API_KEY}",
                json=payload, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    return _unavailable()
                data = await r.json()
                raw  = data["candidates"][0]["content"]["parts"][0]["text"]
                return _parse(raw)
    except Exception as e:
        print(f"  Gemini error: {e}")
        return _unavailable()


def _parse(raw: str) -> Dict:
    clean = re.sub(r'```json?\s*','', raw)
    clean = re.sub(r'```\s*','', clean).strip()
    try:
        p = json.loads(clean)
        fp = float(p.get("fake_probability", 50))
        rp = float(p.get("real_probability", 50))
        total = fp + rp
        if total != 100 and total > 0:
            fp = (fp/total)*100; rp = (rp/total)*100
        return {"available": True, "fake_probability": round(fp,1),
                "real_probability": round(rp,1), "verdict": p.get("verdict","UNCERTAIN"),
                "confidence": p.get("confidence","Moderate"),
                "reasoning": p.get("reasoning",[])[:5],
                "explanation": p.get("explanation",""), "source": "Gemini API"}
    except Exception:
        return _unavailable()


def _unavailable() -> Dict:
    return {"available": False, "fake_probability": 50.0, "real_probability": 50.0,
            "verdict": "UNCERTAIN", "confidence": "N/A",
            "reasoning": ["LLM not configured"], "explanation": "Set GEMINI_API_KEY for AI explanation.",
            "source": "none"}


# ── Rich heuristic rules (used in Full Verify when Gemini not configured) ─
CONSPIRACY_PATTERNS = [
    (r'\b(alien|ufo|extraterrestrial|martian)\b', 30, "Contains alien/UFO claims"),
    (r'\b(illuminati|new world order|deep state|cabal|globalist)\b', 35, "References conspiracy groups"),
    (r'\b(microchip|nanob[oa]t|tracker)\b.{0,30}\b(vaccine|injection|shot)\b', 40, "Vaccine microchip claim"),
    (r'\b(5g|wifi|radiation)\b.{0,30}\b(mind control|cancer|brain|harm)\b', 35, "5G harm claim"),
    (r'\b(cure|heal|treat)\b.{0,20}\b(all|every|any)\b.{0,20}\b(disease|cancer|illness)\b', 38, "Miracle cure claim"),
    (r'\b(government|nasa|cia|who|cdc)\b.{0,30}\b(hiding|suppressing|covering|secret)\b', 32, "Government cover-up claim"),
    (r'\b(chemtrail|chem trail)\b', 35, "Chemtrail claim"),
    (r'\bflat earth\b', 45, "Flat earth claim"),
    (r'\b(lizard|reptilian|shapeshifter)\b', 40, "Reptilian claim"),
    (r'\b(moon landing|lunar landing)\b.{0,20}\b(fake|faked|staged|hoax)\b', 42, "Moon landing hoax"),
    (r'\b(depopulat|eugenics|population control|genocide)\b.{0,30}\b(plan|agenda|plot)\b', 38, "Depopulation conspiracy"),
    (r'\bwhistleblower\b.{0,40}\b(exposes?|reveals?|admits?|confirms?)\b', 20, "Unverified whistleblower claim"),
    (r'\b(share before deleted?|watch before banned|censored)\b', 25, "Censorship pressure tactic"),
    (r'\b(wake up|sheeple|red pill|matrix)\b', 22, "Conspiracy rhetoric"),
    (r'\b(big pharma|pharmaceutical conspiracy)\b', 25, "Big pharma conspiracy"),
]

CREDIBLE_PATTERNS = [
    (r'\b(according to|confirmed by|announced by|stated by)\b', -18, "Cites source"),
    (r'\b(study|research|trial|experiment|survey)\b.{0,30}\b(published|found|shows|reveals|confirms)\b', -20, "Cites research"),
    (r'\b(university|institute|hospital|laboratory|journal)\b', -15, "Academic/medical reference"),
    (r'\b(percent|percentage|basis points?|statistics|data shows)\b', -12, "Uses specific data"),
    (r'\b(official|government|ministry|department|agency)\b.{0,30}\b(announces?|confirms?|approves?|releases?)\b', -18, "Official announcement"),
    (r'\b(reuters|associated press|bbc|bloomberg|times of india|the hindu|ndtv)\b', -22, "Reputable news source"),
    (r'\b(fda|who|cdc|nih|nasa|isro|rbi|sebi)\b.{0,30}\b(approves?|confirms?|finds?|reports?)\b', -25, "Trusted institution"),
    (r'\b(quarterly|annual|fiscal|financial|earnings|gdp|inflation)\b', -15, "Economic reporting"),
    (r'\b(court ruled?|judgment|verdict|legislation|parliament passed?)\b', -18, "Legal/legislative fact"),
    (r'\b(clinical trial|peer.?reviewed|randomized|double.?blind)\b', -25, "Scientific methodology"),
]

def heuristic_llm_fallback(text: str, ml_fake_prob: float, nlp_fake_prob: float) -> Dict:
    """
    Rich rule-based LLM substitute.
    Full Verify uses this with deeper analysis than Fast Check's simple average.
    """
    text_lower = text.lower()
    reasons    = []
    score_adj  = 0
    base_prob  = (ml_fake_prob * 0.6 + nlp_fake_prob * 0.4)

    # Apply conspiracy patterns
    for pattern, weight, label in CONSPIRACY_PATTERNS:
        if re.search(pattern, text_lower):
            score_adj += weight
            reasons.append(f"⚠️ {label}")

    # Apply credibility patterns
    for pattern, weight, label in CREDIBLE_PATTERNS:
        if re.search(pattern, text_lower):
            score_adj += weight  # weight is negative
            reasons.append(f"✅ {label}")

    # Text length analysis
    word_count = len(text.split())
    if word_count < 8:
        score_adj += 15
        reasons.append("⚠️ Very short claim — insufficient context")
    elif word_count > 40:
        score_adj -= 8
        reasons.append("✅ Detailed claim with sufficient context")

    # ALL CAPS check
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.35:
        score_adj += 20
        reasons.append("⚠️ Excessive use of capital letters (common in sensational content)")

    # Question marks & exclamation
    if text.count('!') + text.count('?') >= 3:
        score_adj += 12
        reasons.append("⚠️ Excessive punctuation (common in clickbait)")

    # Hedging language
    if re.search(r'\b(allegedly|reportedly|claimed|rumored|supposedly)\b', text_lower):
        score_adj += 8
        reasons.append("⚠️ Hedging language — unverified claim")

    # Quote marks (may indicate attributed statement)
    if '"' in text or "'" in text:
        score_adj -= 5

    final_fake = max(5, min(95, base_prob + score_adj))
    final_real = 100 - final_fake

    # Build explanation
    if final_fake >= 75:
        verdict = "FAKE"
        conf    = "High"
        expl    = f"This claim shows strong indicators of misinformation ({final_fake:.0f}% fake probability)."
    elif final_fake >= 55:
        verdict = "FAKE"
        conf    = "Moderate"
        expl    = f"This claim leans toward being misinformation ({final_fake:.0f}% fake probability)."
    elif final_fake >= 40:
        verdict = "UNCERTAIN"
        conf    = "Low"
        expl    = f"Mixed signals — cannot confidently classify this claim ({final_fake:.0f}% fake probability)."
    else:
        verdict = "REAL"
        conf    = "High" if final_real >= 75 else "Moderate"
        expl    = f"This claim appears credible ({final_real:.0f}% real probability)."

    top_reasons = [r for r in reasons if r.startswith("⚠️")][:3] + \
                  [r for r in reasons if r.startswith("✅")][:2]
    if not top_reasons:
        top_reasons = ["No strong fake or real indicators detected in language patterns"]

    if score_adj > 20:
        expl += f" Multiple misinformation indicators detected."
    elif score_adj < -20:
        expl += f" Multiple credibility indicators detected."

    return {
        "available":        True,
        "fake_probability": round(final_fake, 1),
        "real_probability": round(final_real, 1),
        "verdict":          verdict,
        "confidence":       conf,
        "reasoning":        top_reasons[:6],
        "explanation":      expl,
        "source":           "Heuristic Analysis (set GEMINI_API_KEY for AI-powered explanation)",
    }
