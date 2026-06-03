"""
Confidence Scoring Service — 50/50 Architecture
================================================
Final score = 50% ML ensemble + 50% Internet (articles + Gemini)

When internet unavailable: falls back to 70% ML + 30% NLP/Heuristic
When Gemini configured:     internet signal is stronger and more accurate
"""
from typing import Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def compute_final_score(ml_result, internet_result, llm_result, nlp_result,
                        fact_result=None) -> Dict:

    ml_fake   = ml_result.get('fake_probability', 50.0)
    ml_avail  = ml_result.get('available', False)
    ml_conf   = ml_result.get('confidence', 50.0)

    net_fake  = internet_result.get('fake_probability', 50.0)
    net_avail = internet_result.get('available', False)
    net_type  = internet_result.get('source_type', 'none')

    nlp_fake  = nlp_result.get('nlp_fake_probability', 50.0)

    # ── Step 1: Fact check hard override ──────────────────────────────────
    # If we KNOW something is factually wrong, boost fake probability
    fact_boost      = 0
    fact_conf_boost = 0
    fact_notes      = []

    if fact_result and fact_result.get('has_fact_error'):
        fact_boost      = fact_result.get('fact_fake_boost', 0)
        fact_conf_boost = fact_result.get('confidence_boost', 0)
        fact_notes      = fact_result.get('contradictions', [])
        ml_fake  = min(95, ml_fake  + fact_boost)
        net_fake = min(90, net_fake + fact_boost * 0.6)
    elif fact_result and fact_result.get('fact_fake_boost', 0) < 0:
        boost    = abs(fact_result.get('fact_fake_boost', 0))
        ml_fake  = max(5, ml_fake  - boost)
        net_fake = max(5, net_fake - boost * 0.5)
        fact_notes = fact_result.get('supports', [])

    # ── Step 2: 50/50 weighting ────────────────────────────────────────────
    # Internet is REAL: means we searched and ran ML on fetched articles
    # + possibly Gemini verified — this is a strong signal
    if net_avail and net_type == 'internet':
        # True 50/50: ML ensemble vs Internet (articles + Gemini)
        w_ml      = 0.50
        w_internet = 0.50
        w_nlp     = 0.0   # NLP already baked into ML
        final_fake = ml_fake * w_ml + net_fake * w_internet

    elif net_avail and net_type == 'local_kb':
        # Local KB available: 55% ML, 30% local KB, 15% NLP
        w_ml      = 0.55
        w_internet = 0.30
        w_nlp     = 0.15
        final_fake = ml_fake * w_ml + net_fake * w_internet + nlp_fake * w_nlp

    else:
        # No internet at all: 65% ML, 35% NLP/heuristic
        w_ml      = 0.65
        w_internet = 0.0
        w_nlp     = 0.35
        final_fake = ml_fake * w_ml + nlp_fake * w_nlp

    # ── Step 3: Gemini direct override ────────────────────────────────────
    # If Gemini returned a confident verdict, give it extra weight
    gemini = internet_result.get('gemini_result')
    if gemini and gemini.get('available') and net_type == 'internet':
        g_fake  = gemini.get('fake_probability', 50.0)
        g_conf  = abs(g_fake - 50)  # 0 = uncertain, 50 = very confident
        if g_conf > 20:
            # Gemini is confident — blend it in directly at 20% extra weight
            final_fake = final_fake * 0.80 + g_fake * 0.20
            fact_conf_boost += int(g_conf * 0.5)

    # ── Step 4: Fact error hard floor ─────────────────────────────────────
    if fact_result and fact_result.get('has_fact_error') and fact_boost >= 35:
        final_fake = max(final_fake, 72.0)

    # ── Step 5: ML high-confidence anchor (no internet) ───────────────────
    # When no internet and ML is very confident, trust it more
    if not net_avail and ml_avail and ml_conf >= 85:
        final_fake = final_fake * 0.30 + ml_fake * 0.70

    final_fake = max(2.0, min(98.0, final_fake))
    final_real = 100.0 - final_fake
    is_fake    = final_fake >= 50.0
    conf       = max(final_fake, final_real)

    if fact_conf_boost > 0:
        conf = min(98.0, conf + fact_conf_boost)

    if conf >= 80:
        conf_label = 'High Confidence'; is_est = False
    elif conf >= 65:
        conf_label = 'Moderate Confidence'; is_est = False
    else:
        conf_label = 'Low Confidence'; is_est = True

    sens        = nlp_result.get('sensationalism', {})
    sens_score  = sens.get('score', 0)
    sens_level  = sens.get('level', 'Low')

    # ── Breakdown for frontend ─────────────────────────────────────────────
    gemini_disp = None
    if gemini and gemini.get('available'):
        gemini_disp = {
            'fake_probability': gemini.get('fake_probability', 50),
            'verdict':   gemini.get('verdict', 'N/A'),
            'reasoning': gemini.get('reasoning', ''),
        }

    ml_on_art = internet_result.get('ml_on_articles')
    breakdown = {
        'ml_model': {
            'fake_probability': round(ml_fake, 1),
            'real_probability': round(100 - ml_fake, 1),
            'weight':     '50%' if (net_avail and net_type == 'internet') else '65%',
            'available':  ml_avail,
            'confidence': round(ml_conf, 1),
        },
        'internet_verification': {
            'fake_probability': round(net_fake, 1),
            'real_probability': round(100 - net_fake, 1),
            'weight':       '50%' if (net_avail and net_type == 'internet') else ('30%' if net_avail else '0%'),
            'available':    net_avail,
            'source_type':  net_type,
            'sources_count':internet_result.get('sources_count', 0),
            'supporting':   internet_result.get('supporting', 0),
            'contradicting':internet_result.get('contradicting', 0),
            'ml_on_articles': ml_on_art,
            'gemini':       gemini_disp,
        },
        'fact_check': {
            'available':      bool(fact_result),
            'has_error':      fact_result.get('has_fact_error', False) if fact_result else False,
            'contradictions': fact_notes[:3],
            'boost_applied':  fact_boost,
        },
        'nlp_signals': {
            'fake_probability':     round(nlp_fake, 1),
            'real_probability':     round(100 - nlp_fake, 1),
            'weight':               '0%' if (net_avail and net_type == 'internet') else '35%',
            'sensationalism':       sens_level,
            'sensationalism_score': sens_score,
            'sentiment':            nlp_result.get('sentiment', {}).get('tone', 'N/A'),
            'detected_signals':     sens.get('detected', []),
            'credibility_signals':  sens.get('credibility_signals', 0),
        },
    }

    explanation = _build_explanation(
        is_fake, final_fake, fact_result, net_avail, net_type,
        internet_result, gemini, nlp_result, is_est
    )

    # Collect reasoning
    reasoning = []
    if fact_notes:
        reasoning.extend([f'Fact error: {n}' for n in fact_notes[:2]])
    if gemini and gemini.get('reasoning'):
        reasoning.append(f'Gemini: {gemini["reasoning"]}')
    if gemini and gemini.get('key_facts'):
        reasoning.extend(gemini['key_facts'][:2])
    if ml_on_art and ml_on_art.get('available'):
        n = ml_on_art.get('articles_analyzed', 0)
        reasoning.append(f'ML analyzed {n} internet articles: {ml_on_art["fake_probability"]:.0f}% fake')
    sens_detected = sens.get('detected', [])
    if sens_detected:
        reasoning.append(f'Sensational language: {", ".join(sens_detected[:3])}')

    return {
        'verdict':          'FAKE' if is_fake else 'REAL',
        'is_fake':          is_fake,
        'confidence':       round(conf, 1),
        'confidence_label': conf_label,
        'is_estimated':     is_est,
        'fake_probability': round(final_fake, 1),
        'real_probability': round(final_real, 1),
        'breakdown':        breakdown,
        'explanation':      explanation,
        'reasoning':        reasoning,
        'language':         nlp_result.get('language', 'en'),
        'sources':          internet_result.get('sources', []),
        'fact_notes':       fact_notes,
    }


def _build_explanation(is_fake, fake_prob, fact_result, net_avail, net_type,
                        internet_result, gemini, nlp_result, is_est) -> str:
    parts = []
    real_prob = 100 - fake_prob

    # Fact error takes top priority
    if fact_result and fact_result.get('has_fact_error'):
        contras = fact_result.get('contradictions', [])
        if contras:
            parts.append(f'Factual error detected: {contras[0]}.')

    # Gemini explanation (most readable)
    if gemini and gemini.get('available') and gemini.get('reasoning'):
        parts.append(gemini['reasoning'])

    if is_fake:
        parts.append(f'Classified as fake news ({fake_prob:.0f}% probability).')
        contra = internet_result.get('contradicting', 0)
        total  = internet_result.get('sources_count', 0)
        if net_avail and net_type == 'internet':
            if contra > 0:
                parts.append(f'{contra}/{total} internet sources contradict this claim.')
            elif total > 0:
                parts.append(f'No sources found confirming this claim.')
        sens = nlp_result.get('sensationalism', {})
        if sens.get('level') in ['High', 'Medium'] and sens.get('detected'):
            parts.append(f'Sensational language: {", ".join(sens["detected"][:2])}.')
    else:
        parts.append(f'Classified as real news ({real_prob:.0f}% probability).')
        supp  = internet_result.get('supporting', 0)
        total = internet_result.get('sources_count', 0)
        if net_avail and net_type == 'internet' and supp > 0:
            parts.append(f'{supp}/{total} sources support this claim.')

    if is_est:
        parts.append('Confidence is low — treat as an estimate.')

    return ' '.join(parts)
