from pathlib import Path
import json
import pandas as pd
from src.utils import ROOT

def summarize_ocr(rows):
    """rows: iterable of dicts with ground_truth and prediction keys."""
    rows = list(rows)
    if not rows:
        return {"count": 0, "exact_match_accuracy": None, "character_accuracy": None}
    exact = sum(str(r["ground_truth"]).upper() == str(r["prediction"]).upper() for r in rows)
    total_chars = sum(len(str(r["ground_truth"])) for r in rows)
    char_errors = 0
    for r in rows:
        a, b = str(r["ground_truth"]).upper(), str(r["prediction"]).upper()
        # Levenshtein distance with a small dynamic-programming implementation.
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
            prev = cur
        char_errors += prev[-1]
    return {"count": len(rows), "exact_match_accuracy": exact / len(rows),
            "character_accuracy": max(0.0, 1 - char_errors / max(total_chars, 1))}

def save_report(report, filename="evaluation.json"):
    out = ROOT / "outputs/reports" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
