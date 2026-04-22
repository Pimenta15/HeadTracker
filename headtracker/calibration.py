"""Persist and load neutral-pose calibration data."""
import json
import os


def load(path):
    """Return calibration dict from JSON file, or None if missing/corrupt."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibração salva em {path}")
