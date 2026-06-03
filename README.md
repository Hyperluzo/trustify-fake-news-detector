# Trustify
AI based news checker 
# 🛡️ Trustify — AI-Powered Fake News Detection System

> Minor Project 

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25-brightgreen)]()


| Gaurav Maurya 



---

## 🚀 What is Trustify?

Trustify is an AI-powered web application that detects whether a news headline
or claim is **Real ✅ or Fake ❌** using a hybrid approach combining:

- **4-Algorithm ML Ensemble** (Logistic Regression + Random Forest + XGBoost + Passive Aggressive)
- **Fact Checker** with world knowledge (leaders, capitals, science facts, history)
- **Internet Verification** — searches top articles and runs ML on them
- **Gemini AI** for natural language explanation
- **50/50 Scoring** — 50% ML + 50% Internet in Full Verify mode

---

## ✅ What it can detect

| Input | Result |
|---|---|
| "Trump is PM of India" | ❌ FAKE 98% |
| "ChatGPT is made by Google" | ❌ FAKE 87% |
| "Sun revolves around the Earth" | ❌ FAKE 87% |
| "Aliens landed in Delhi" | ❌ FAKE 98% |
| "Vaccines cause autism" | ❌ FAKE 98% |
| "Modi is PM of India" | ✅ REAL 98% |
| "RBI holds repo rate 6.5%" | ✅ REAL 82% |
| "Federal Reserve raises rates" | ✅ REAL 83% |

---

## 📁 Project Structure
trustify_v3/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── config.py            # Settings & API keys
│   ├── train_ensemble.py    # ML training pipeline
│   └── services/
│       ├── nlp_service.py       # NLP + ML prediction
│       ├── fact_checker.py      # World knowledge facts
│       ├── search_service.py    # Internet verification
│       ├── llm_service.py       # Gemini AI integration
│       └── scoring_service.py   # 50/50 weighted scoring
├── frontend/
│   └── index.html           # Complete web UI
├── requirements.txt
└── run.py                   # One-click launcher

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/trustify-fake-news-detector.git
cd trustify-fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
cd backend
python train_ensemble.py
```

### 4. Start the server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open browser
http://localhost:8000

---

## 🤖 Algorithms Used

| Algorithm | Role | Weight |
|---|---|---|
| Logistic Regression | Baseline classifier | ×2 |
| Random Forest | Non-linear patterns | ×2 |
| XGBoost | Gradient boosting | ×3 |
| Passive Aggressive | Text specialist | ×1 |
| **Soft Voting** | Combined ensemble | — |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 95%+ (with Kaggle data) |
| Fact Check Accuracy | 25/25 (100%) |
| Response Time (Fast) | ~200ms |
| Response Time (Full) | 3-8 seconds |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Fast ML prediction |
| POST | `/api/verify` | Full internet + AI check |
| GET | `/api/health` | Server status |
| GET | `/api/history` | Past analyses |

---

## 🔑 Optional API Keys

Add in `backend/config.py` for enhanced accuracy:
- **Gemini API** — free at https://aistudio.google.com
- **NewsAPI** — free at https://newsapi.org
- **SerpAPI** — free at https://serpapi.com

---

## 🔮 Future Improvements

- BERT/DistilBERT for 99% accuracy
- Mobile app (React Native)
- Browser extension
- Multi-language Hindi support
- Real-time social media monitoring

---

## 📄 License

MIT License — free to use for educational purposes.
