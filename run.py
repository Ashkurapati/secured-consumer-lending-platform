"""
run.py — Stepping Stones entry point
Run from the project root:
    python run.py
"""
import uvicorn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "website"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("ENV", "development") != "production"
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=reload)
