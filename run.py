"""
run.py — Stepping Stones entry point
Run from the project root:
    python run.py
"""
import uvicorn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "website"))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
