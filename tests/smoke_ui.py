"""Headless end-to-end UI test.  Run: xvfb-run -a python tests/smoke_ui.py
Default: FAKE engine + synthetic leaf (tests UI/data flow only).  USE_REAL_MODEL=1: real model in model/
(flow only - accuracy is measured by training/evaluate.py and tests/validate_real_model.py)."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KIVY_NO_ARGS", "1")
import numpy as np
from PIL import Image, ImageDraw
from kivy.clock import Clock

from services.app_context import AppContext
from services.prediction_service import MODEL_DIR, PredictionService
from main import PlantCareApp

TMP = tempfile.mkdtemp()
OUT = os.environ.get("SMOKE_OUT", "/tmp")
KEYS = ["early_blight", "late_blight", "healthy"]


class Fixed:
    num_classes = 3
    def predict_proba(self, b): return np.tile(np.array([0.9, 0.06, 0.04], np.float32), (len(b), 1))


def make_ctx():
    if os.environ.get("USE_REAL_MODEL") == "1":
        return AppContext(os.path.join(TMP, "data"))
    md = os.path.join(TMP, "model"); os.makedirs(md)
    json.dump({"plant": "tomato", "version": "fake", "engine": "numpy", "model_file": "m.npz", "input_size": 16,
               "normalization": "zero_1", "resize_mode": "squash", "class_keys": KEYS,
               "class_labels": {"early_blight": "Early blight", "late_blight": "Late blight", "healthy": "Healthy"},
               "thresholds": {"min_confidence": 0.6, "min_margin": 0.15}}, open(os.path.join(md, "model_config.json"), "w"))
    ctx = AppContext(os.path.join(TMP, "data"), model_dir=md)
    ctx.predictor = PredictionService(md, engine_override=Fixed()); ctx.model_error = None
    return ctx


leaf = os.path.join(TMP, "leaf.jpg")
im = Image.new("RGB", (640, 480), (90, 70, 50)); d = ImageDraw.Draw(im); d.ellipse((100, 40, 540, 440), fill=(40, 140, 50))
for i in range(60): d.ellipse((150 + i * 6, 150 + (i * 37) % 200, 160 + i * 6, 160 + (i * 37) % 200), fill=(110, 80, 30))
im.save(leaf)
if os.environ.get("LEAF_IMAGE"):
    leaf = os.environ["LEAF_IMAGE"]   # e.g. a real dataset test image
bad = os.path.join(TMP, "bad.jpg"); open(bad, "wb").write(b"junk" * 50)

ctx = make_ctx()
app = PlantCareApp(ctx=ctx)
results = []
chk = lambda n, ok: results.append((n, bool(ok)))
state = {"pick": leaf}
fake_pick = lambda cb: cb(state["pick"], None)
shot = lambda name: app.root.export_to_png(os.path.join(OUT, name))


def wait_for(cond, then, tries=100):
    if cond() or tries == 0:
        then(); return
    Clock.schedule_once(lambda dt: wait_for(cond, then, tries - 1), 0.1)


def start(dt):
    scan = app.scan; scan.pick_fn = fake_pick
    shot("ui_home.png")
    chk("home shows empty history", ctx.db.count() == 0)
    app.home.btn_upload.dispatch("on_release")
    Clock.schedule_once(lambda dt: step_analyze(scan), 0.8)


def step_analyze(scan):
    chk("upload -> scan + preview", app.sm.current == "scan" and scan.image_path)
    state["pick"] = bad; scan.btn_gallery.dispatch("on_release")
    chk("corrupt image rejected, preview kept", "corrupted" in scan.error_label.text and scan.image_path)
    scan.btn_analyze.dispatch("on_release")
    chk("busy state shown during analysis", scan.btn_analyze.text == "Analyzing..." or app.sm.current == "result")
    wait_for(lambda: app.sm.current == "result", step_result)


def step_result():
    r = app.result.record
    chk("analysis opened result screen", app.sm.current == "result" and r)
    chk("result saved to SQLite", r and r["saved"] and ctx.db.count() == 1)
    chk("result has label + valid confidence", r["predicted_label"] and 0 <= r["confidence"] <= 1)
    chk("scan screen reset after analysis", app.scan.image_path is None and app.scan.btn_remove.disabled)
    chk("saved image still exists", os.path.isfile(r["image_path"]))
    Clock.schedule_once(lambda dt: (shot("ui_result.png"), step_history()), 0.4)


def step_history():
    app.sm.current = "history"
    Clock.schedule_once(lambda dt: step_history2(), 0.4)


def step_history2():
    from ui.rows import HistoryRow
    rows = [w for w in app.sm.get_screen("history").col.children if isinstance(w, HistoryRow)]
    chk("history lists the saved scan", len(rows) == 1)
    shot("ui_history.png")
    app.open_record(rows[0].record, "history")
    chk("history item opens details", app.sm.current == "result" and app.result.record["id"] == rows[0].record["id"])
    app.sm.current = "learn"
    learn = app.sm.get_screen("learn"); learn.search.text = "whitefly"
    from ui.rows import SimpleRow
    hits = [w for w in learn.col.children if isinstance(w, SimpleRow)]
    chk("learn search finds the whitefly-spread virus", len(hits) == 1 and "curl" in hits[0].children[-1].text.lower())
    shot("ui_learn.png")
    app.open_disease(ctx.info.get("tomato", "late_blight"))
    chk("disease detail opens", app.sm.current == "disease")
    # privacy: do not save
    ctx.settings.set("save_history", False)
    app.sm.current = "scan"; state["pick"] = leaf; app.scan.btn_gallery.dispatch("on_release")
    app.scan.btn_analyze.dispatch("on_release")
    wait_for(lambda: app.sm.current == "result", step_unsaved)


def step_unsaved():
    r = app.result.record
    chk("save_history off -> not stored", r and not r["saved"] and ctx.db.count() == 1)
    tmp = r["image_path"]; chk("temp image exists while viewing", os.path.isfile(tmp))
    app.sm.current = "home"
    Clock.schedule_once(lambda dt: step_cleanup(tmp), 0.4)


def step_cleanup(tmp):
    chk("unsaved temp image removed on leave", not os.path.exists(tmp))
    rid = ctx.db.list()[0]["id"]; path = ctx.db.list()[0]["image_path"]
    app.sm.get_screen("history")._do_delete(rid)
    chk("delete removes record + image", ctx.db.count() == 0 and not os.path.exists(path))
    app.sm.current = "settings"; app.sm.current = "about"
    chk("settings/about screens build", True)
    # esc/back navigation
    app._on_key(None, 27); chk("back key leaves About", app.sm.current == "home")
    for n, ok in results: print(("PASS " if ok else "FAIL ") + n)
    app.stop(); sys.exit(0 if all(ok for _, ok in results) else 1)


Clock.schedule_once(start, 1.0)
app.run()
