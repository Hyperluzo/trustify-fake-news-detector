"""
NLP Service — Sensationalism, Clickbait, Sentiment, Language Detection
Supports English + Hindi
"""
import re
import pickle
import os
import numpy as np
from typing import Tuple, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Sensationalism keywords (English) ──────────────────────────────────────
SENSATIONAL_EN = [
    "BREAKING","BOMBSHELL","SHOCKING","SECRET","EXPOSED","REVEALED","EXCLUSIVE",
    "COVER-UP","CONSPIRACY","BANNED","CENSORED","SILENCED","URGENT","SUPPRESSED",
    "THEY DON'T WANT","MAINSTREAM MEDIA","WHISTLEBLOWER","SHARE BEFORE DELETED",
    "WAKE UP","WHAT THEY HIDE","TRUTH THEY","BIG PHARMA","DEEP STATE",
    "FALSE FLAG","NEW WORLD ORDER","ILLUMINATI","GOVERNMENT HIDING",
    "LEAKED DOCUMENT","ANONYMOUS SOURCE","INSIDER REVEALS",
]

# ── Hindi sensational keywords (Devanagari) ────────────────────────────────
SENSATIONAL_HI = [
    "सरकार छुपा रही", "बड़ा खुलासा", "चौंकाने वाला", "सच्चाई सामने",
    "मीडिया नहीं बताएगी", "वायरल सच", "षड्यंत्र", "धमाकेदार खुलासा",
    "सरकार का झूठ", "अफवाह", "फेक न्यूज़",
]

# ── Credibility indicators (real news signals) ─────────────────────────────
CREDIBLE_SIGNALS = [
    "according to","study published","researchers found","fda approved",
    "supreme court","congress passed","officially confirmed","press conference",
    "peer reviewed","clinical trial","university study","official statement",
    "report shows","statistics show","data reveals","survey conducted",
    "spokesperson said","minister announced","court ruled","agency confirmed",
    "percent of","basis points","quarterly earnings","annual report",
    "reuters","associated press","afp","who said","un report",
]

# ── Clickbait patterns ──────────────────────────────────────────────────────
CLICKBAIT_PATTERNS = [
    r"you won'?t believe",
    r"what (they|doctors|scientists) don'?t want",
    r"(doctors|scientists) hate (him|her|this)",
    r"number \d+ will shock",
    r"this one (weird|simple) trick",
    r"share before (this is |it gets )?deleted",
    r"they are hiding",
    r"the truth about",
    r"what mainstream media",
    r"exposed!|revealed!|confirmed!",
    r"\d+% of (people|doctors|scientists) (don'?t|won'?t)",
]

STOPWORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her','it','its',
    'they','them','their','what','who','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','do','does','did','a','an',
    'the','and','but','if','or','as','at','by','for','with','of','to','from','in',
    'out','on','off','up','down','all','both','each','some','no','not','so','than',
    'very','can','will','just','now','then','when','where','how','which','there',
    'here','would','could','may','might',
}

def simple_stem(word: str) -> str:
    for s in ['ing','tion','tions','ness','ment','ers','ed','ly','ies','ize',
              'ful','less','able','ible','er','es']:
        if word.endswith(s) and len(word)-len(s) >= 4:
            return word[:-len(s)]
    return word

def preprocess(text: str) -> str:
    """Full NLP pipeline used at inference time — must match training."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(simple_stem(t) for t in text.split()
                    if t not in STOPWORDS and len(t) > 2)

def detect_language(text: str) -> str:
    """Detect if text is Hindi (Devanagari) or English."""
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    return 'hi' if hindi_chars > 3 else 'en'

def translate_hindi_to_english(text: str) -> str:
    """
    Basic Hindi → English transliteration for ML processing.
    In production: use Google Translate API or IndicTrans.
    This heuristic handles common patterns for offline operation.
    """
    # Common Hindi fake news phrases mapped to English equivalents
    replacements = {
        'सरकार': 'government', 'छुपा': 'hiding', 'सच': 'truth',
        'झूठ': 'lie fake', 'अफवाह': 'rumor fake', 'खुलासा': 'exposed secret',
        'षड्यंत्र': 'conspiracy', 'वायरल': 'viral', 'दावा': 'claim',
        'जांच': 'investigation', 'रिपोर्ट': 'report', 'सूत्र': 'source',
        'अनुसार': 'according', 'पुष्टि': 'confirmed', 'आरोप': 'allegation',
    }
    for hindi, english in replacements.items():
        text = text.replace(hindi, english)
    # Remove remaining Devanagari (keep any transliterated parts)
    text = re.sub(r'[\u0900-\u097F]+', ' ', text)
    return text.strip() or "unverified claim"

def sensationalism_score(text: str) -> Dict:
    """
    Compute sensationalism score 0-100.
    Returns score + breakdown of detected signals.
    """
    text_upper = text.upper()
    text_lower = text.lower()

    detected = []

    # Check sensational keywords
    sens_hits = [kw for kw in SENSATIONAL_EN if kw in text_upper]
    detected.extend(sens_hits[:5])

    # Check clickbait patterns
    click_hits = [p for p in CLICKBAIT_PATTERNS if re.search(p, text_lower)]
    detected.extend([f"clickbait:{p[:20]}" for p in click_hits[:3]])

    # Excessive punctuation / caps
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.3:
        detected.append("EXCESSIVE_CAPS")

    exclaim = text.count('!') + text.count('?')
    if exclaim >= 2:
        detected.append(f"EXCESSIVE_PUNCTUATION({exclaim})")

    # Credibility signals (reduce fake score)
    cred_hits = sum(1 for s in CREDIBLE_SIGNALS if s in text_lower)

    # Raw score
    raw = len(sens_hits) * 15 + len(click_hits) * 20
    if caps_ratio > 0.3: raw += 15
    if exclaim >= 2: raw += 10
    raw = max(0, raw - cred_hits * 10)
    score = min(100, raw)

    return {
        "score": score,
        "level": "High" if score >= 60 else "Medium" if score >= 30 else "Low",
        "detected": detected[:8],
        "credibility_signals": cred_hits,
    }

def sentiment_score(text: str) -> Dict:
    """
    Simple lexicon-based sentiment analysis.
    Returns polarity and emotional tone.
    """
    positive_words = {
        'good','great','excellent','best','amazing','wonderful','fantastic',
        'success','win','positive','improve','benefit','hope','relief','safe',
        'confirmed','approved','proven','official','legitimate','verified',
    }
    negative_words = {
        'bad','evil','terrible','worst','horrible','awful','dangerous','deadly',
        'kill','death','destroy','conspiracy','lie','fake','fraud','corrupt',
        'secret','hidden','expose','shocking','bombshell','urgent','crisis',
    }

    tokens = text.lower().split()
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    total = max(pos + neg, 1)

    polarity = (pos - neg) / total  # -1 (very negative) to +1 (very positive)

    if polarity > 0.2:
        tone = "Positive"
    elif polarity < -0.2:
        tone = "Negative"
    else:
        tone = "Neutral"

    return {
        "polarity": round(polarity, 3),
        "tone": tone,
        "positive_signals": pos,
        "negative_signals": neg,
    }

def keyword_score(text: str) -> Dict:
    """Score text based on fake vs real news keyword presence."""
    text_lower = text.lower()

    fake_keywords = [
        'conspiracy','secret','expose','cover up','cover-up','hidden truth',
        'they don\'t want','suppressed','illuminati','new world order',
        'deep state','big pharma','chemtrail','microchip','5g','plandemic',
        'hoax','false flag','staged','crisis actor','woke','globalist',
        'satanic','ritual','clone','shapeshifter','reptilian','alien',
        'vaccine kill','poison water','fluoride','mind control',
    ]
    real_keywords = [
        'study','research','published','journal','university','professor',
        'official','government','ministry','department','spokesperson',
        'confirmed','verified','investigation','report','data','statistics',
        'percent','survey','poll','trial','experiment','evidence','analysis',
        'according to','source said','announced','released','approved',
    ]

    fake_hits = [kw for kw in fake_keywords if kw in text_lower]
    real_hits = [kw for kw in real_keywords if kw in text_lower]

    fake_score = min(len(fake_hits) * 12, 100)
    real_score = min(len(real_hits) * 8, 100)

    return {
        "fake_keyword_score": fake_score,
        "real_keyword_score": real_score,
        "fake_keywords_found": fake_hits[:5],
        "real_keywords_found": real_hits[:5],
        "net_fake_bias": fake_score - real_score,
    }

def full_nlp_analysis(text: str) -> Dict:
    """Run all NLP analyses and return combined result."""
    lang    = detect_language(text)
    process_text = translate_hindi_to_english(text) if lang == 'hi' else text

    sens    = sensationalism_score(process_text)
    sent    = sentiment_score(process_text)
    kw      = keyword_score(process_text)

    # NLP fake probability (0-100, used as 10% weight in final score)
    nlp_fake_prob = min(100, (
        sens['score'] * 0.4 +
        max(0, kw['net_fake_bias']) * 0.4 +
        max(0, -sent['polarity'] * 50) * 0.2
    ))

    return {
        "language": lang,
        "sensationalism": sens,
        "sentiment": sent,
        "keywords": kw,
        "nlp_fake_probability": round(nlp_fake_prob, 1),
        "nlp_real_probability": round(100 - nlp_fake_prob, 1),
        "preprocessed_text": preprocess(process_text),
    }


class MLPredictor:
    """Loads and runs the trained ensemble model."""

    def __init__(self):
        self.model      = None
        self.vectorizer = None
        self.metadata   = {}
        self._load()

    def _load(self):
        model_path = os.path.join(BASE_DIR, 'models', 'ensemble_model.pkl')
        vect_path  = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')
        meta_path  = os.path.join(BASE_DIR, 'models', 'model_metadata.pkl')

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Ensemble model loaded")
        else:
            print("⚠️  Ensemble model not found. Run: python backend/train_ensemble.py")

        if os.path.exists(vect_path):
            with open(vect_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("✅ TF-IDF vectorizer loaded")

        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Metadata loaded — Accuracy: {self.metadata.get('accuracy')}%")

    def predict(self, text: str) -> Dict:
        """Return ML prediction with probabilities."""
        if self.model is None or self.vectorizer is None:
            return {
                "fake_probability": 50.0,
                "real_probability": 50.0,
                "confidence": 50.0,
                "available": False,
            }

        lang = detect_language(text)
        process_text = translate_hindi_to_english(text) if lang == 'hi' else text
        clean = preprocess(process_text)
        vec   = self.vectorizer.transform([clean])
        prob  = self.model.predict_proba(vec)[0]  # [fake_prob, real_prob]

        return {
            "fake_probability": round(float(prob[0]) * 100, 1),
            "real_probability": round(float(prob[1]) * 100, 1),
            "confidence":       round(max(prob) * 100, 1),
            "available":        True,
        }

# Singleton instance
ml_predictor = MLPredictor()
