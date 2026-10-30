"""End-to-end check of the INSTALLED model through the app's own pipeline (PredictionService + image_service)
on the held-out test split of the dataset. Not a unit test (needs the dataset).
python tests/validate_real_model.py --data /path/to/PlantVillage/raw/color"""
import argparse, collections, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.prediction_service import PredictionService

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
a = ap.parse_args()
svc = PredictionService()
rows = [r for r in csv.DictReader(open("training/reports/split_manifest.csv")) if r["split"] == "test"]
n = ok = acc_n = acc_ok = flagged_leaf = 0; per = collections.Counter(); t0 = time.time()
for r in rows:
    p = svc.predict(os.path.join(a.data, r["file"]))
    n += 1; hit = p.top_key == r["class"]; ok += hit
    if p.status == "ok": acc_n += 1; acc_ok += hit
    if any("does not look like a leaf" in w for w in p.warnings): flagged_leaf += 1
out = {"images": n, "accuracy_all": round(ok / n, 4), "answered_ok": acc_n, "accuracy_when_status_ok": round(acc_ok / max(acc_n, 1), 4),
       "inconclusive_share": round(1 - acc_n / n, 4), "flagged_not_leaf_heuristic": flagged_leaf,
       "ms_per_image_incl_decode": round((time.time() - t0) / n * 1000, 1)}
print(json.dumps(out, indent=1)); json.dump(out, open("training/reports/app_pipeline_validation.json", "w"), indent=1)
