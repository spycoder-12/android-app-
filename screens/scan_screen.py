"""Scan screen: pick/capture a leaf image, preview it, validate it, replace/remove it."""
import os

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.screenmanager import Screen
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView

from services import image_service, platform_service
from ui import theme
from ui.widgets import Card, PrimaryButton, SecondaryButton, SectionTitle, WrapLabel


class ScanScreen(Screen):
    def __init__(self, ctx=None, pick_fn=None, capture_fn=None, storage_dir=None, **kwargs):
        super().__init__(name="scan", **kwargs)
        # Injectable so tests can replace the OS-level pickers.
        self.pick_fn = pick_fn or platform_service.pick_image
        self.capture_fn = capture_fn or platform_service.capture_photo
        self.ctx = ctx
        self._storage_dir = storage_dir or (ctx.images_dir if ctx else None)
        self._busy_ev = None
        self.image_path = None   # managed, upright JPEG copy used for preview and analysis
        self.analyze_handler = None  # set in Phase 5

        with self.canvas.before:
            Color(*theme.CREAM)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._sync_bg, size=self._sync_bg)

        scroll = ScrollView(do_scroll_x=False)
        col = BoxLayout(orientation="vertical", size_hint_y=None, padding=dp(theme.PAD),
                        spacing=dp(theme.GAP))
        col.bind(minimum_height=col.setter("height"))
        scroll.add_widget(col)
        self.add_widget(scroll)

        col.add_widget(SectionTitle(text="Scan a leaf"))

        card = Card(size_hint_y=None, height=dp(300))
        self.preview = KivyImage(allow_stretch=True, keep_ratio=True, opacity=0)
        self.empty_label = WrapLabel(
            text="No image selected.\nChoose a photo from your gallery or take a new one.",
            halign="center", valign="middle", color=theme.TEXT_MUTED, size_hint_y=1)
        self.empty_label.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.holder = BoxLayout()
        self.holder.add_widget(self.empty_label)
        card.add_widget(self.holder)
        col.add_widget(card)

        self.info_label = WrapLabel(text="", font_size=sp(13), color=theme.TEXT_MUTED)
        self.warn_label = WrapLabel(text="", font_size=sp(13), color=theme.BROWN)
        self.error_label = WrapLabel(text="", font_size=sp(14), color=theme.ERROR, bold=True)
        for w in (self.info_label, self.warn_label, self.error_label):
            col.add_widget(w)

        self.busy_bar = ProgressBar(max=100, value=0, size_hint_y=None, height=0, opacity=0)
        col.add_widget(self.busy_bar)

        self.btn_gallery = SecondaryButton(text="Choose from gallery")
        self.btn_camera = SecondaryButton(text="Take photo")
        self.btn_remove = SecondaryButton(text="Remove image", disabled=True)
        self.btn_analyze = PrimaryButton(text="Analyze leaf")
        for b in (self.btn_gallery, self.btn_camera, self.btn_remove, self.btn_analyze):
            col.add_widget(b)
        self.btn_gallery.bind(on_release=lambda *_: self.choose_from_gallery())
        self.btn_camera.bind(on_release=lambda *_: self.take_photo())
        self.btn_remove.bind(on_release=lambda *_: self.clear_image())
        self.btn_analyze.bind(on_release=lambda *_: self.analyze())

        col.add_widget(SectionTitle(text="Photo tips", font_size=sp(16)))
        col.add_widget(WrapLabel(
            text="- Use daylight; avoid harsh shadows\n- Fill the frame with one leaf\n"
                 "- Keep the leaf in focus and flat\n- Use a plain background if possible",
            font_size=sp(13)))

    # -- lifecycle ------------------------------------------------------
    def _sync_bg(self, *_):
        self._bg.pos, self._bg.size = self.pos, self.size

    def on_pre_enter(self, *_):
        mode = getattr(self.manager, "scan_mode", None)
        if self.manager is not None:
            self.manager.scan_mode = None
        if mode == "upload":
            Clock.schedule_once(lambda _dt: self.choose_from_gallery(), 0.2)
        elif mode == "capture":
            Clock.schedule_once(lambda _dt: self.take_photo(), 0.2)

    @property
    def storage_dir(self):
        if self._storage_dir:
            return self._storage_dir
        app = App.get_running_app()
        return os.path.join(app.user_data_dir if app else ".", "user_images")

    # -- actions --------------------------------------------------------
    def choose_from_gallery(self):
        self._set_error("")
        self.pick_fn(self._on_source_result)

    def take_photo(self):
        self._set_error("")
        tmp = os.path.join(self.storage_dir, "tmp_capture.jpg")
        self.capture_fn(tmp, self._on_source_result)

    def _on_source_result(self, path, error=None):
        if error:
            self._set_error(error)
        elif path:
            self.set_image(path)
        # path None and no error = user cancelled; do nothing

    def set_image(self, src_path):
        """Validate `src_path`, make a managed upright copy, and show the preview."""
        try:
            managed = image_service.save_managed_copy(src_path, self.storage_dir)
            img = image_service.load_image(managed)
        except image_service.ImageError as exc:
            self._set_error(str(exc))
            return False
        except OSError:
            self._set_error("Could not read or store this image. Check free storage and try again.")
            return False
        self._discard_current()
        self.image_path = managed
        self.preview.source = managed
        self.preview.reload()
        self.holder.clear_widgets()
        self.holder.add_widget(self.preview)
        self.preview.opacity = 1
        self.info_label.text = f"{os.path.basename(src_path)}  |  {img.size[0]}x{img.size[1]} px"
        warns = image_service.assess_quality(img) if (not self.ctx or self.ctx.settings.get("show_quality_hints")) else []
        self.warn_label.text = ("Quality hints (heuristic): " + " ".join(warns)) if warns else ""
        self._set_error("")
        self.btn_remove.disabled = False
        self.btn_gallery.text = "Replace from gallery"
        self.btn_camera.text = "Replace with new photo"
        return True

    def clear_image(self):
        self._discard_current()
        self.image_path = None
        self.preview.opacity = 0
        self.holder.clear_widgets()
        self.holder.add_widget(self.empty_label)
        self.info_label.text = self.warn_label.text = ""
        self._set_error("")
        self.btn_remove.disabled = True
        self.btn_gallery.text = "Choose from gallery"
        self.btn_camera.text = "Take photo"

    def analyze(self):
        try:
            image_service.validate_image(self.image_path)
        except image_service.ImageError as exc:
            self._set_error(str(exc))
            return
        if self.analyze_handler is None:
            self._set_error("Analysis is not available yet: no trained model is installed "
                            "(added in Phases 4-5).")
            return
        self.analyze_handler(self.image_path)

    def set_busy(self, busy):
        """Loading state while the model runs (buttons locked, progress bar animating)."""
        for b in (self.btn_gallery, self.btn_camera, self.btn_remove, self.btn_analyze):
            b.disabled = busy or (b is self.btn_remove and self.image_path is None)
        self.btn_analyze.text = "Analyzing..." if busy else "Analyze leaf"
        self.busy_bar.height = dp(10) if busy else 0
        self.busy_bar.opacity = 1 if busy else 0
        if self._busy_ev:
            self._busy_ev.cancel(); self._busy_ev = None
        if busy:
            self._set_error("")
            self._busy_ev = Clock.schedule_interval(
                lambda _dt: setattr(self.busy_bar, "value", (self.busy_bar.value + 7) % 100), 0.05)

    def release_image(self):
        """Hand the stored image over to history/result without deleting it; reset the screen."""
        self.image_path = None
        self.clear_image()

    # -- helpers --------------------------------------------------------
    def _discard_current(self):
        if self.image_path and os.path.isfile(self.image_path):
            try:
                os.remove(self.image_path)
            except OSError:
                pass

    def _set_error(self, msg):
        self.error_label.text = msg
