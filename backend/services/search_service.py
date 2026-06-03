"""
Search + Verify Service — New Architecture
==========================================
Step 1: Search internet for top articles about the claim
Step 2: Run ML model on each fetched article
Step 3: Also check claim against local knowledge base
Step 4: Return combined internet signal

Supports: DuckDuckGo (free), NewsAPI (optional), SerpAPI (optional), Gemini (optional)
"""
import asyncio, aiohttp, re, os, sys
from typing import List, Dict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# ── Known misinformation topics (local KB) ────────────────────────────────
KNOWN_FAKE_TOPICS = [
    "aliens landed","moon landing faked","flat earth","hollow earth","lizard people",
    "microchip vaccine","5g mind control","bill gates microchip","chemtrail","illuminati",
    "new world order","deep state controls","george soros pays","qanon","crisis actor",
    "plandemic","covid engineered","bleach cure","ivermectin cures covid",
    "big pharma hiding","doctors hiding cure","natural cure cancer",
    "government hiding truth","suppressed technology","free energy suppressed",
    "time travel portal","bigfoot captured","portal to hell","world leaders clones",
    "celebrity faked death","satanic hollywood","rothschild controls","bitcoin cia",
    "population control","depopulation agenda","water fluoride iq",
    "microwave toxic","sunscreen cancer","wifi brain tumor",
    "urine therapy","raw onion prevents","ancient herb reverses",
    "election rigged","voting machine hacked","cia september 11","false flag attack",
    "weather machine","haarp earthquake","planet x nibiru","second moon hidden",
    "nasa cover up","government cloning","mk ultra active","facebook recording",
]

KNOWN_REAL_TOPICS = [
    "nasa james webb","federal reserve raises","supreme court ruled","who declares",
    "fda approves","clinical trial","peer reviewed study","scientists discover",
    "researchers found","official statement","press conference","quarterly earnings",
    "unemployment rate","inflation falls","gdp grows","central bank",
    "election results certified","parliament passes","court ruling","ministry announces",
    "isro launches","chandrayaan","spacex rocket","cancer survival rate","vaccine efficacy",
    "earthquake magnitude","hurricane category","wildfire contained",
    "species population increase","renewable energy record","electric vehicle sales",
    "rbi repo rate","sensex record","india gdp","reserve bank",
    "budget announced","tax reform","infrastructure project",
    "elon musk","twitter x","openai","chatgpt","google deepmind",
    "sachin tendulkar","virat kohli","ms dhoni","rohit sharma",
    "ukraine russia","nato","un security council","world bank imf",
]

CREDIBLE_DOMAINS = [
    "bbc.","reuters.","apnews.","nytimes.","theguardian.","bloomberg.",
    "thehindu.","ndtv.","timesofindia.","indianexpress.","hindustantimes.",
    "livemint.","economictimes.","scroll.in","thewire.in",
    "who.int","cdc.gov","nasa.gov","isro.gov.in","rbi.org.in","wikipedia.org",
    "britannica.com","snopes.com","factcheck.org","altnews.in","boomlive.in",
]

REFUTE_WORDS = [
    'false','fake','hoax','misleading','misinformation','debunk',
    'fact-check','no evidence','unverified','conspiracy','rumor',
    'not true','incorrect','inaccurate','disproven','fabricated',
    'satire','parody','claim is false',
]

SUPPORT_WORDS = [
    'confirmed','verified','official','report','study','research',
    'published','according to','scientists','government','university',
    'hospital','data','evidence','analysis','investigation','approved',
    'announced','results show','statistics','percent','survey',
]


def extract_keywords(text: str) -> str:
    """Pull most meaningful words for search query."""
    stopwords = {
        'is','are','was','were','the','a','an','in','on','at','to','for',
        'or','and','but','real','fake','true','false','this','that','has',
        'have','had','been','being','with','from','by','as','it','its',
    }
    tokens = re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    return ' '.join(tokens[:8])


# ── Internet search functions ──────────────────────────────────────────────

async def search_duckduckgo(query: str, session) -> List[Dict]:
    """DuckDuckGo Instant Answer — free, no key needed."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; Trustify/2.0)'}
        params  = {'q': query, 'format': 'json', 'no_html': '1',
                   'no_redirect': '1', 'skip_disambig': '1'}
        async with session.get(
            'https://api.duckduckgo.com/', params=params, headers=headers,
            timeout=aiohttp.ClientTimeout(total=6)
        ) as r:
            if r.status == 200:
                d = await r.json(content_type=None)
                results = []
                abstract = d.get('Abstract', '').strip()
                if abstract:
                    results.append({
                        'title': d.get('Heading', query),
                        'content': abstract[:500],
                        'url': d.get('AbstractURL', ''),
                        'source': d.get('AbstractSource', 'DuckDuckGo'),
                    })
                for topic in d.get('RelatedTopics', [])[:5]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append({
                            'title': topic.get('Text', '')[:100],
                            'content': topic.get('Text', '')[:400],
                            'url': topic.get('FirstURL', ''),
                            'source': 'DuckDuckGo',
                        })
                return results
    except Exception as e:
        print(f'  DDG error: {type(e).__name__}')
    return []


async def search_newsapi(query: str, session) -> List[Dict]:
    """NewsAPI — 500 req/day free tier."""
    if not config.NEWS_API_KEY:
        return []
    try:
        params = {
            'q': query[:100], 'apiKey': config.NEWS_API_KEY,
            'pageSize': 5, 'sortBy': 'relevancy', 'language': 'en',
        }
        async with session.get(
            'https://newsapi.org/v2/everything', params=params,
            timeout=aiohttp.ClientTimeout(total=8)
        ) as r:
            if r.status == 200:
                data = await r.json()
                results = []
                for a in data.get('articles', [])[:5]:
                    content = ' '.join(filter(None, [
                        a.get('title', ''),
                        a.get('description', ''),
                        a.get('content', '')[:300],
                    ]))
                    results.append({
                        'title':   a.get('title', ''),
                        'content': content[:600],
                        'url':     a.get('url', ''),
                        'source':  a.get('source', {}).get('name', 'NewsAPI'),
                    })
                print(f'  NewsAPI: {len(results)} articles fetched')
                return results
    except Exception as e:
        print(f'  NewsAPI error: {e}')
    return []


async def search_serpapi(query: str, session) -> List[Dict]:
    """SerpAPI Google search — 100/month free."""
    if not config.SERP_API_KEY:
        return []
    try:
        params = {
            'q': query, 'api_key': config.SERP_API_KEY,
            'num': 5, 'engine': 'google',
        }
        async with session.get(
            'https://serpapi.com/search', params=params,
            timeout=aiohttp.ClientTimeout(total=8)
        ) as r:
            if r.status == 200:
                data = await r.json()
                results = []
                for item in data.get('organic_results', [])[:5]:
                    results.append({
                        'title':   item.get('title', ''),
                        'content': item.get('snippet', '')[:400],
                        'url':     item.get('link', ''),
                        'source':  'Google Search',
                    })
                print(f'  SerpAPI: {len(results)} results fetched')
                return results
    except Exception as e:
        print(f'  SerpAPI error: {e}')
    return []


async def search_gemini(query: str, original_claim: str, session) -> Dict:
    """
    Ask Gemini to fact-check the claim directly.
    Returns a structured verdict with probability.
    """
    if not config.GEMINI_API_KEY:
        return None

    prompt = f"""You are a fact-checker. Analyze this claim and respond ONLY in JSON:

Claim: "{original_claim}"

Search context: "{query}"

Respond ONLY with this JSON (no markdown, no extra text):
{{
  "fake_probability": <number 0-100>,
  "real_probability": <number 0-100>,
  "verdict": "FAKE" or "REAL" or "UNCERTAIN",
  "reasoning": "1-2 sentence explanation of why",
  "key_facts": ["fact1", "fact2"]
}}

Guidelines:
- Be factual and objective
- Conspiracy theories, miracle cures, alien stories = FAKE
- Factual errors (wrong leader, wrong country, wrong date) = FAKE
- News from credible sources with verifiable facts = REAL
- If you are unsure, say UNCERTAIN with 50/50 probability"""

    url     = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={config.GEMINI_API_KEY}'
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': 0.1, 'maxOutputTokens': 300},
    }
    try:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=12)
        ) as r:
            if r.status == 200:
                data = await r.json()
                raw  = data['candidates'][0]['content']['parts'][0]['text']
                import json as _json
                clean = re.sub(r'```json?\s*', '', raw)
                clean = re.sub(r'```\s*', '', clean).strip()
                parsed = _json.loads(clean)
                fp = float(parsed.get('fake_probability', 50))
                rp = float(parsed.get('real_probability', 50))
                total = fp + rp
                if total > 0 and total != 100:
                    fp = (fp/total)*100; rp = (rp/total)*100
                print(f'  Gemini: {parsed.get("verdict")} ({fp:.0f}% fake)')
                return {
                    'available':        True,
                    'fake_probability': round(fp, 1),
                    'real_probability': round(rp, 1),
                    'verdict':          parsed.get('verdict', 'UNCERTAIN'),
                    'reasoning':        parsed.get('reasoning', ''),
                    'key_facts':        parsed.get('key_facts', []),
                }
            else:
                print(f'  Gemini API error: {r.status}')
    except Exception as e:
        print(f'  Gemini error: {type(e).__name__}: {str(e)[:80]}')
    return None


# ── ML on fetched content ─────────────────────────────────────────────────

def run_ml_on_articles(articles: List[Dict], ml_predictor) -> Dict:
    """
    Run the ML ensemble on each fetched article's content.
    Articles about fake topics will score high fake%.
    Articles confirming real news will score high real%.
    Returns averaged result across all articles.
    """
    if not articles or ml_predictor is None or ml_predictor.model is None:
        return {'available': False, 'fake_probability': 50.0, 'real_probability': 50.0}

    from services.nlp_service import preprocess
    fake_scores = []
    real_scores = []

    for article in articles[:5]:
        content = article.get('content', '') + ' ' + article.get('title', '')
        if len(content.strip()) < 20:
            continue
        try:
            clean = preprocess(content)
            vec   = ml_predictor.vectorizer.transform([clean])
            prob  = ml_predictor.model.predict_proba(vec)[0]
            fake_scores.append(float(prob[0]) * 100)
            real_scores.append(float(prob[1]) * 100)
        except Exception:
            pass

    if not fake_scores:
        return {'available': False, 'fake_probability': 50.0, 'real_probability': 50.0}

    avg_fake = sum(fake_scores) / len(fake_scores)
    avg_real = sum(real_scores) / len(real_scores)

    return {
        'available':        True,
        'fake_probability': round(avg_fake, 1),
        'real_probability': round(avg_real, 1),
        'articles_analyzed': len(fake_scores),
    }


# ── Source credibility analysis ────────────────────────────────────────────

def analyze_source_credibility(articles: List[Dict], original_claim: str) -> Dict:
    """
    Analyze each fetched article to see if it supports or contradicts the claim.
    Checks domain credibility + content signals.
    """
    if not articles:
        return {'supporting': 0, 'contradicting': 0, 'neutral': 0, 'source_details': []}

    claim_lower  = original_claim.lower()
    claim_tokens = set(re.sub(r'[^a-z\s]', '', claim_lower).split())

    supporting = contradicting = neutral = 0
    source_details = []

    for a in articles:
        combined = (a.get('title', '') + ' ' + a.get('content', '')).lower()
        url      = a.get('url', '').lower()

        refute_ct  = sum(1 for w in REFUTE_WORDS  if w in combined)
        support_ct = sum(1 for w in SUPPORT_WORDS if w in combined)

        # Domain credibility bonus
        is_credible = any(d in url for d in CREDIBLE_DOMAINS)
        if is_credible:
            support_ct += 2

        # Overlap with claim
        src_tokens = set(re.sub(r'[^a-z\s]', '', combined).split())
        overlap    = len(claim_tokens & src_tokens) / max(len(claim_tokens), 1)

        if refute_ct >= 2 and refute_ct > support_ct:
            verdict = 'Contradicts'; contradicting += 1
        elif support_ct > refute_ct and overlap > 0.15:
            verdict = 'Supports'; supporting += 1
        else:
            verdict = 'Neutral'; neutral += 1

        source_details.append({
            'title':   a.get('title', '')[:100],
            'source':  a.get('source', 'Unknown'),
            'url':     a.get('url', ''),
            'verdict': verdict,
        })

    return {
        'supporting':    supporting,
        'contradicting': contradicting,
        'neutral':       neutral,
        'source_details': source_details,
    }


# ── Local knowledge base (offline fallback) ────────────────────────────────

def local_kb_check(text: str) -> Dict:
    """Check text against curated local knowledge base when internet unavailable."""
    text_lower = text.lower()

    fake_hits = [t for t in KNOWN_FAKE_TOPICS if t in text_lower]
    real_hits = [t for t in KNOWN_REAL_TOPICS if t in text_lower]

    fake_score = min(len(fake_hits) * 28, 85)
    real_score = min(len(real_hits) * 22, 85)

    total = fake_score + real_score
    if total == 0:
        return {
            'available': True, 'source_type': 'local_kb',
            'fake_probability': 52.0, 'real_probability': 48.0,
            'sources_count': 0, 'supporting': 0, 'contradicting': 0,
            'neutral': 0, 'sources': [],
            'explanation': 'No matching patterns in knowledge base.',
            'ml_on_articles': None, 'gemini_result': None,
        }

    fp = (fake_score / total) * 100
    rp = 100 - fp

    pseudo = []
    for m in fake_hits[:3]:
        pseudo.append({'title': f"Known misinformation pattern: '{m}'",
                       'source': 'Knowledge Base', 'url': '', 'verdict': 'Contradicts'})
    for m in real_hits[:3]:
        pseudo.append({'title': f"Credible topic: '{m}'",
                       'source': 'Knowledge Base', 'url': '', 'verdict': 'Supports'})

    return {
        'available': True, 'source_type': 'local_kb',
        'fake_probability': round(fp, 1), 'real_probability': round(rp, 1),
        'sources_count': len(pseudo),
        'supporting': len(real_hits), 'contradicting': len(fake_hits),
        'neutral': 0, 'sources': pseudo,
        'explanation': f"Matched {len(fake_hits)} fake + {len(real_hits)} real patterns.",
        'ml_on_articles': None, 'gemini_result': None,
    }


# ── MAIN: Search → ML on articles → Gemini → Combine ─────────────────────

async def verify_claim_online(text: str, ml_predictor_ref=None) -> Dict:
    """
    New pipeline:
    1. Search internet for top articles on this topic
    2. Run ML ensemble on those articles
    3. Ask Gemini to fact-check (if key configured)
    4. Combine internet ML + Gemini → internet signal
    5. Return structured result for scoring_service
    """
    query = extract_keywords(text)
    if len(query) < 5:
        query = text[:80]

    print(f'  Searching: "{query}"')

    # Lazy import to avoid circular
    if ml_predictor_ref is None:
        try:
            from services.nlp_service import ml_predictor as _mlp
            ml_predictor_ref = _mlp
        except Exception:
            pass

    all_articles: List[Dict] = []
    gemini_result = None

    async with aiohttp.ClientSession() as session:
        # Run all searches + Gemini concurrently
        tasks = [
            search_duckduckgo(query, session),
            search_newsapi(query, session),
            search_serpapi(query, session),
            search_gemini(query, text, session),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate article results from Gemini result
    for r in results[:3]:
        if isinstance(r, list):
            all_articles.extend(r)
    gemini_raw = results[3]
    if isinstance(gemini_raw, dict):
        gemini_result = gemini_raw

    # Deduplicate articles
    seen, unique = set(), []
    for a in all_articles:
        key = a.get('url', '') or a.get('title', '')
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    unique = unique[:8]  # top 8 articles

    print(f'  Found {len(unique)} unique articles | Gemini: {"✅" if gemini_result else "❌ (no key)"}')

    # If no internet results at all → local KB
    if not unique and not gemini_result:
        return local_kb_check(text)

    # Step 2: Run ML on fetched articles
    ml_on_articles = run_ml_on_articles(unique, ml_predictor_ref)

    # Step 3: Analyze source credibility
    cred = analyze_source_credibility(unique, text)

    # Step 4: Combine signals into internet fake probability
    # Base from source credibility
    total_src = len(unique)
    if total_src > 0:
        contra_ratio = cred['contradicting'] / total_src
        support_ratio = cred['supporting'] / total_src
        cred_fake = contra_ratio * 70 + (1 - support_ratio) * 30
    else:
        cred_fake = 52.0

    # Blend with ML-on-articles (if available)
    if ml_on_articles['available']:
        internet_fake = (cred_fake * 0.40 + ml_on_articles['fake_probability'] * 0.60)
    else:
        internet_fake = cred_fake

    # Blend with Gemini (if available) — Gemini gets strong say
    if gemini_result and gemini_result.get('available'):
        internet_fake = (internet_fake * 0.40 + gemini_result['fake_probability'] * 0.60)

    internet_fake = max(5.0, min(95.0, internet_fake))
    internet_real = 100.0 - internet_fake

    # Build explanation
    parts = []
    if gemini_result and gemini_result.get('reasoning'):
        parts.append(gemini_result['reasoning'])
    if cred['contradicting'] > 0:
        parts.append(f"{cred['contradicting']}/{total_src} sources contradict this claim.")
    elif cred['supporting'] > 0:
        parts.append(f"{cred['supporting']}/{total_src} sources support this claim.")
    if ml_on_articles['available']:
        n = ml_on_articles['articles_analyzed']
        parts.append(f"ML analysis on {n} fetched articles: {ml_on_articles['fake_probability']:.0f}% fake.")
    explanation = ' '.join(parts) or 'Internet search completed.'

    return {
        'available':         True,
        'source_type':       'internet',
        'fake_probability':  round(internet_fake, 1),
        'real_probability':  round(internet_real, 1),
        'sources_count':     total_src,
        'supporting':        cred['supporting'],
        'contradicting':     cred['contradicting'],
        'neutral':           cred['neutral'],
        'sources':           cred['source_details'],
        'explanation':       explanation,
        'ml_on_articles':    ml_on_articles,
        'gemini_result':     gemini_result,
    }
