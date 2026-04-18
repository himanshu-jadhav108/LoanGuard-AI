"""Compatibility entrypoint for legacy Streamlit app."""

from pathlib import Path
import runpy
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

runpy.run_module("loanguard.apps.app", run_name="__main__")
