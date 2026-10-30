# PlantCare AI

Android-ready tomato leaf disease helper built with Python + Kivy. Analysis runs **on the device** (pure NumPy
inference, no TensorFlow in the app); photos and history never leave the phone.

## What exists and what has been verified

| Item | Status |
|---|---|
| Kivy UI (Home, Scan, Result, Learn, History, Settings, About) | Built; headless UI test passes 19/19 (fake engine and real model) |
| Gallery pick / camera capture | Desktop picker tested headlessly with fakes. **Android gallery + camera: NOT tested on a phone** |
| Image validation / preprocessing | 18 unit tests |
| SQLite history, disease info JSON, prediction logic | 50+ unit tests total (`pytest -q`) |
| Trained model | Yes: compact CNN trained from scratch on PlantVillage tomato (10 classes) |
| APK | **Not built.** Build it on your Linux/WSL machine (below) and test on a phone |

## Model results (held-out test split, 2698 images, leaf-grouped split)

Accuracy **94.5%**, macro-F1 **0.936**. Through the app's own pipeline: 94.48%
(identical); on results not flagged "inconclusive" (92.3% of images) accuracy is 97.5%.
Inconclusive threshold: score < 0.77 (chosen on the validation split only) or top-2 margin < 0.15.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bacterial_spot | 0.916 | 0.968 | 0.942 | 316 |
| early_blight | 0.926 | 0.932 | 0.929 | 148 |
| healthy | 0.840 | 1.000 | 0.913 | 237 |
| late_blight | 0.975 | 0.961 | 0.968 | 284 |
| leaf_mold | 0.957 | 0.957 | 0.957 | 140 |
| mosaic_virus | 0.930 | 0.982 | 0.955 | 54 |
| septoria_leaf_spot | 1.000 | 0.908 | 0.952 | 260 |
| spider_mites | 0.904 | 0.915 | 0.910 | 248 |
| target_spot | 0.879 | 0.833 | 0.856 | 210 |
| yellow_leaf_curl_virus | 1.000 | 0.961 | 0.980 | 801 |

**Read these limits before trusting it**
- Data are clean PlantVillage lab photos. **No real field photos were evaluated**; expect lower real-world accuracy.
- Simulated degradation accuracy on the test set: darker 64%, brighter 89%, blur 79%, crop 89%, noise 19%. The model is **very sensitive to image noise**
  (a few phone photos in low light may mislead it). Retrain with `--backbone mobilenetv2` on a PC/Colab
  (transfer learning, needs ImageNet weights) and add noise/JPEG augmentation to improve robustness.
- Softmax scores are not calibrated probabilities and cannot reliably detect non-tomato or non-leaf images.
  A colour heuristic and the threshold above add some protection only.
- Leaf grouping comes from the dataset's leaf-map where available (~60% of images); the rest use an approximate
  filename-block heuristic, so a small amount of leakage between splits is possible.
- Export checks: NumPy engine vs Keras max difference 1.9e-06
  (argmax agreement 100%); TFLite vs Keras likewise. Speed on a laptop-class CPU
  is a few ms/image; **inference time on a real phone has not been measured**.
- Not a diagnosis. Disease text is general education; it names no pesticides or doses.

## 1. Run on Windows (VS Code)

```powershell
cd PlantCareAI
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt     # Python 3.10-3.12 recommended for Kivy 2.3.1 wheels
python main.py
```
Pick "Upload Leaf Image" -> choose a JPG/PNG -> "Analyze leaf". (Camera capture works on Android only.)

Tests: `pytest -q`  |  Headless UI test (Linux/WSL): `xvfb-run -a python tests/smoke_ui.py`
(`USE_REAL_MODEL=1` and `LEAF_IMAGE=/path/to/leaf.jpg` use the real model).

## 2. Reproduce the training (optional)

```bash
pip install -r requirements-train.txt
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/spMohanty/PlantVillage-Dataset pv
cd pv && git sparse-checkout init --no-cone && git sparse-checkout set README.md CITATION.cff leaf-map.json 'raw/color/Tomato___*' && git checkout && cd ..
python training/prepare_data.py --data pv/raw/color --leaf-map pv/leaf-map.json
python training/train.py --epochs 16            # or --backbone mobilenetv2 (needs internet for ImageNet weights)
python training/evaluate.py
python training/convert_model.py                # writes model/ (npz, tflite, model_config.json)
python tests/validate_real_model.py --data pv/raw/color
```
**Dataset license:** the repo has no license file. Verify the terms on the dataset's Hugging Face card before
redistributing data or models. Cite Mohanty, Hughes & Salathe (2016). Do not commit the dataset.
Reports are in `training/reports/` (metrics, confusion matrix, split manifest, export validation).

## 3. Build the APK (Ubuntu / WSL2)

```bash
sudo apt update && sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config \
  zlib1g-dev libncurses-dev cmake libffi-dev libssl-dev libltdl-dev
pip install --user buildozer cython
cd PlantCareAI && buildozer android debug     # first build downloads SDK/NDK; 20-60 min
# APK appears in bin/
```
Install: enable USB debugging, `adb install -r bin/*.apk` (or copy the APK to the phone and open it).
Check on the phone: gallery pick, camera, analysis, history, delete, airplane-mode analysis, back button.

### Troubleshooting
- `kivy==2.3.1` recipe not found -> try `kivy==2.3.0` in `buildozer.spec` or set `p4a.branch = develop`.
- SDK/NDK/license errors -> `buildozer android clean`, keep `android.ndk = 25b`, `android.accept_sdk_license = True`.
- Build in WSL's Linux filesystem (`~/`), not `/mnt/c/...`.
- **Gallery returns no file / permission error on Android 10+**: plyer may not resolve `content://` URIs. Next step is the
  `androidstorage4kivy` library (copy the picked file into app storage).
- **Camera fails on Android 10+**: the plyer camera intent writes via a file path, which modern Android restricts.
  The code uses the app's external files dir and `StrictMode.disableDeathOnFileUriExposure()`, which may not work on
  every version. Workaround: take the photo with the camera app and use "Choose from gallery"; proper fix: a
  FileProvider-based intent or `camera4kivy`.
- App closes at start -> `adb logcat -s python` and send the traceback.

## Layout
`main.py` | `screens/` | `ui/` | `services/` (image, prediction, NumPy engine, info, settings, platform, context) |
`database/` | `data/` (disease + tips JSON) | `model/` (trained artifacts + config) | `training/` | `tests/`
