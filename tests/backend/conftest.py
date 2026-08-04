"""Shared pytest fixtures.

Adds backend/ to sys.path so tests can `import services.xxx` the same
way the app code does, and points storage paths at a temp directory so
tests never touch real indexed data.
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
