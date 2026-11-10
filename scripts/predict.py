#!/usr/bin/env python3
"""Predict image class and save ELA and heuristic suspicious-region visuals."""
import argparse, json, sys
from pathlib import Path
import cv2, joblib, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from image_forensics.features import load_image, extract_features, ela_map, suspicious_heatmap

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path); p.add_argument("--model", type=Path, default=ROOT/"models/random_forest.joblib")
    p.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    a=p.parse_args()
    if not a.model.exists(): raise SystemExit(f"Model not found: {a.model}. Run scripts/train.py first.")
    rgb=load_image(a.image); bundle=joblib.load(a.model); model=bundle["model"]
    x=extract_features(rgb).reshape(1,-1); pred=str(model.predict(x)[0]); probs=model.predict_proba(x)[0]
    classes=list(model.classes_); confidence=float(probs[classes.index(pred)])
    a.output_dir.mkdir(parents=True,exist_ok=True); stem=a.image.stem
    ela_path=a.output_dir/f"{stem}_ela.png"; heat_path=a.output_dir/f"{stem}_heatmap.png"
    cv2.imwrite(str(ela_path),cv2.cvtColor(ela_map(rgb),cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heat_path),suspicious_heatmap(rgb))
    result={"image":str(a.image),"prediction":pred,"confidence":confidence,
            "probabilities":{str(c):float(v) for c,v in zip(classes,probs)},
            "heatmap":str(heat_path),"ela_visualization":str(ela_path),
            "heatmap_note":"Heuristic ELA contrast visualization, not a validated localization mask."}
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
