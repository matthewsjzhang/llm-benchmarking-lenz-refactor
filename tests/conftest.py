import os
import sys

# Make `from utils import helper_inference` work like it does in src/00_call_models.py
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
