from pathlib import Path
import cv2
import pandas as pd
from src.utils import ROOT

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def audit_images(folder):
    folder = Path(folder)
    rows = []
    if not folder.exists():
        return pd.DataFrame(columns=["file", "valid", "width", "height", "bytes"])
    for p in folder.rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        im = cv2.imread(str(p))
        rows.append({"file": str(p), "valid": im is not None,
                     "width": int(im.shape[1]) if im is not None else None,
                     "height": int(im.shape[0]) if im is not None else None,
                     "bytes": p.stat().st_size})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = ROOT / "outputs/reports/dataset_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = audit_images(ROOT / "data/raw/vehicles")
    df.to_csv(out, index=False)
    print(f"Images audited: {len(df)}; invalid: {int((~df['valid']).sum()) if len(df) else 0}; report: {out}")
