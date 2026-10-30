[app]
title = PlantCare AI
package.name = plantcareai
package.domain = org.plantcare
source.dir = .
source.include_exts = py,png,jpg,json,npz,kv,atlas
source.include_patterns = model/*,data/*,assets/*
source.exclude_dirs = tests,training,.venv,bin,.buildozer,__pycache__,dataset,user_images,docs
source.exclude_patterns = *.keras,*.h5,*.tflite,*.db,*.md
version = 1.0.0

# numpy + pillow have python-for-android recipes. No TensorFlow: inference uses the pure-NumPy engine.
requirements = python3,kivy==2.3.1,numpy,pillow,plyer,pyjnius,android

orientation = portrait
fullscreen = 0

android.permissions = CAMERA,READ_EXTERNAL_STORAGE,READ_MEDIA_IMAGES
android.api = 34
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
android.allow_backup = False

[buildozer]
log_level = 2
warn_on_root = 1
