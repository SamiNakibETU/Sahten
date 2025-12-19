"""
Sahten - Vercel Serverless Entry Point
======================================

This file is the entry point for Vercel's Python serverless functions.
It imports and exposes the FastAPI application from the backend.

Environment Variables (set in Vercel dashboard):
  - OPENAI_API_KEY: Required for LLM features
  - UPSTASH_REDIS_REST_URL: Optional, for persistent logging
  - UPSTASH_REDIS_REST_TOKEN: Optional, for persistent logging
"""

import sys
from pathlib import Path

# Add the backend directory to Python path
# This allows imports like "from app.bot import SahtenBot"
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Import the FastAPI app from main.py
from main import app

# Vercel expects the app to be available at module level
# The "app" variable is automatically picked up by @vercel/python

