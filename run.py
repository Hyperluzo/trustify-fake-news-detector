"""
=============================================================
  Trustify v2.0 — Single Entry Point
  Authors: Chandan Upadhyay | Gaurav Maurya | Harshvardhan Vaishnav
  JECRC University, Jaipur | 2025-26

  Usage:
    python run.py               → start on localhost:8000
    python run.py --public      → start + ngrok public URL
    python run.py --train       → retrain model first, then start
    python run.py --port 9000   → custom port
=============================================================
"""

import os, sys, subprocess, time, argparse, socket

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "ensemble_model.pkl")

# ── Argument Parser ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Trustify v2.0 Launcher")
parser.add_argument("--public", action="store_true", help="Create public URL via ngrok")
parser.add_argument("--train",  action="store_true", help="Retrain model before starting")
parser.add_argument("--port",   type=int, default=8000, help="Port number (default: 8000)")
args = parser.parse_args()

def banner():
    print("\n" + "═"*60)
    print("  🛡️  Trustify v2.0 — AI Fake News Detection System")
    print("  Chandan Upadhyay | Gaurav Maurya | Harshvardhan Vaishnav")
    print("  JECRC University, Jaipur | 2025-26")
    print("═"*60)

def check_model():
    if not os.path.exists(MODEL_PATH):
        print("\n  ⚠️  Model not found. Training now (first-time setup)...")
        train_model()
    else:
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"  ✅ Ensemble model found ({size_mb:.1f} MB)")

def train_model():
    train_script = os.path.join(BACKEND_DIR, "train_ensemble.py")
    result = subprocess.run([sys.executable, train_script], cwd=BASE_DIR)
    if result.returncode != 0:
        print("  ❌ Training failed. Check backend/train_ensemble.py")
        sys.exit(1)
    print("  ✅ Model trained successfully!")

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def start_with_ngrok(port):
    """Start FastAPI + create public ngrok tunnel."""
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("\n  Installing pyngrok...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "-q"])
        from pyngrok import ngrok, conf

    # Check for token
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token:
        print("\n" + "─"*55)
        print("  📋 NGROK SETUP (one-time only)")
        print("─"*55)
        print("  1. Go to https://ngrok.com → Sign up FREE")
        print("  2. Dashboard → Your Authtoken → Copy it")
        print("─"*55)
        token = input("  Paste your ngrok authtoken: ").strip()
        if not token:
            print("  ⚠️  No token — falling back to local only.")
            start_local(port)
            return

    ngrok.set_auth_token(token)

    # Start FastAPI in background
    import threading
    import uvicorn

    sys.path.insert(0, BACKEND_DIR)

    def run_server():
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="warning",
        )

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(3)  # Wait for FastAPI to boot

    # Open tunnel
    print("\n  🌐 Creating public URL tunnel...")
    try:
        tunnel    = ngrok.connect(port, "http")
        public_url = tunnel.public_url.replace("http://", "https://")
    except Exception as e:
        print(f"  ❌ ngrok error: {e}")
        start_local(port)
        return

    print("\n" + "═"*60)
    print("  🛡️  TRUSTIFY IS LIVE!")
    print("═"*60)
    print(f"\n  🌍 PUBLIC URL (anyone, anywhere):")
    print(f"     {public_url}")
    print(f"\n  💻 Local:")
    print(f"     http://localhost:{port}")
    print(f"\n  📚 API Docs:  {public_url}/docs")
    print(f"\n  📱 Works on:")
    print(f"     ✅ Any laptop on any network")
    print(f"     ✅ Mobile phones")
    print(f"     ✅ Anyone with the link worldwide")
    print(f"\n  Press Ctrl+C to stop")
    print("═"*60 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  🛑 Shutting down...")
        ngrok.kill()

def start_local(port):
    """Start FastAPI locally and show network info."""
    local_ip = get_local_ip()

    print(f"\n  ✅ Starting FastAPI server on port {port}...")
    print("\n" + "═"*60)
    print("  🛡️  TRUSTIFY IS RUNNING")
    print("═"*60)
    print(f"\n  This device:       http://localhost:{port}")
    print(f"  Same WiFi devices: http://{local_ip}:{port}")
    print(f"  API Docs:          http://localhost:{port}/docs")
    print(f"\n  Endpoints:")
    print(f"    POST /api/predict  — Fast ML-only prediction")
    print(f"    POST /api/verify   — Full: ML + Internet + LLM")
    print(f"    POST /api/explain  — Detailed explanation")
    print(f"    GET  /api/history  — Analysis history")
    print(f"    GET  /api/health   — Server status")
    print(f"\n  Tip: Run with --public for a shareable public URL")
    print(f"  Press Ctrl+C to stop")
    print("═"*60 + "\n")

    # Start uvicorn
    sys.path.insert(0, BACKEND_DIR)
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=[BACKEND_DIR],
        log_level="info",
    )

def main():
    banner()

    # Step 1: Train if requested or model missing
    if args.train:
        print("\n  [1/2] Training ensemble model...")
        train_model()
    else:
        print("\n  [1/2] Checking model...")
        check_model()

    # Step 2: Launch
    print(f"\n  [2/2] Launching server on port {args.port}...")
    if args.public:
        start_with_ngrok(args.port)
    else:
        start_local(args.port)

if __name__ == "__main__":
    main()
