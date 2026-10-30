"""Platform glue: gallery picking, camera capture and Android runtime permissions.

Desktop: Kivy file-chooser popup; camera is not supported (reported honestly).
Android: plyer file chooser / camera + runtime permissions.
!! The Android paths cannot be exercised on a desktop - they must be tested on a real phone.
"""
import os
from kivy.clock import Clock
from kivy.utils import platform

IS_ANDROID = platform == "android"


def request_permissions(kind, callback):
    """kind: 'camera' | 'gallery'. callback(granted: bool) runs on the Kivy thread."""
    if not IS_ANDROID:
        callback(True)
        return
    from android.permissions import Permission, request_permissions as _req  # noqa
    if kind == "camera":
        perms = [Permission.CAMERA]
    else:
        # Android 13+ uses READ_MEDIA_IMAGES; older uses READ_EXTERNAL_STORAGE.
        perms = [getattr(Permission, "READ_MEDIA_IMAGES", None) or Permission.READ_EXTERNAL_STORAGE,
                 Permission.READ_EXTERNAL_STORAGE]
    _req(perms, lambda _p, grants: Clock.schedule_once(lambda _dt: callback(any(grants)), 0))


def pick_image(on_result):
    """Ask the user for an image. on_result(path_or_None, error_or_None)."""
    if IS_ANDROID:
        def after_perm(ok):
            if not ok:
                on_result(None, "Gallery permission was denied. Enable it in Settings > Apps > PlantCare AI.")
                return
            try:
                from plyer import filechooser
                filechooser.open_file(
                    on_selection=lambda sel: Clock.schedule_once(
                        lambda _dt: on_result(sel[0] if sel else None, None), 0),
                    filters=[["Images", "*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]],
                    multiple=False)
            except Exception as exc:  # noqa
                on_result(None, f"Could not open the gallery: {exc}")
        request_permissions("gallery", after_perm)
    else:
        from ui.file_picker import open_desktop_picker
        open_desktop_picker(lambda path: on_result(path, None))


def _android_capture_path(default_path):
    """The camera app cannot write into our private storage; use the app's external files dir."""
    try:
        from jnius import autoclass
        act = autoclass("org.kivy.android.PythonActivity").mActivity
        d = act.getExternalFilesDir(None)
        if d is not None:
            return os.path.join(d.getAbsolutePath(), "capture.jpg")
    except Exception:
        pass
    return default_path


def capture_photo(dest_path, on_result):
    """Take a photo with the phone camera, saved to dest_path. on_result(path_or_None, error_or_None)."""
    if not IS_ANDROID:
        on_result(None, "Camera capture is only available on Android. "
                        "On a computer, use 'Choose from gallery' instead.")
        return

    def after_perm(ok):
        if not ok:
            on_result(None, "Camera permission was denied. Enable it in Settings > Apps > PlantCare AI.")
            return
        try:
            from plyer import camera
            out = _android_capture_path(dest_path)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            try:   # needed on Android 7+ so the camera intent may use a file:// path (may not work on all versions)
                from jnius import autoclass
                autoclass("android.os.StrictMode").disableDeathOnFileUriExposure()
            except Exception:
                pass

            def done(path):
                Clock.schedule_once(lambda _dt: on_result(
                    path if path and os.path.isfile(path) else None,
                    None if path and os.path.isfile(path) else "No photo was captured."), 0)
            camera.take_picture(filename=out, on_complete=done)
        except Exception as exc:  # noqa
            on_result(None, f"Could not open the camera: {exc}")
    request_permissions("camera", after_perm)
