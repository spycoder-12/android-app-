"""PlantCare AI - application entry point."""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, SlideTransition
from kivy.utils import platform

from screens.about_screen import AboutScreen
from screens.history_screen import HistoryScreen
from screens.home_screen import HomeScreen
from screens.learn_screen import DiseaseDetailScreen, LearnScreen
from screens.result_screen import ResultScreen
from screens.scan_screen import ScanScreen
from screens.settings_screen import SettingsScreen
from services.app_context import AppContext
from services.image_service import ImageError
from services.prediction_service import ModelError
from ui import theme

NAV_ITEMS = [("home", "Home"), ("learn", "Learn"), ("history", "History"),
             ("settings", "Settings"), ("about", "About")]
BACK_MAP = {"scan": "home", "learn": "home", "history": "home", "settings": "home", "about": "home",
            "disease": "learn"}


class NavBar(BoxLayout):
    def __init__(self, sm, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None, height=dp(56), **kwargs)
        self.sm = sm
        self.buttons = {}
        for key, label in NAV_ITEMS:
            b = Button(text=label, font_size=sp(12), background_normal="", background_down="",
                       background_color=theme.WHITE, color=theme.TEXT_MUTED)
            b.bind(on_release=lambda _b, k=key: self.go(k))
            self.buttons[key] = b
            self.add_widget(b)
        sm.bind(current=self._highlight)
        self._highlight(sm, sm.current)

    def go(self, key):
        self.sm.current = key

    def _highlight(self, _sm, current):
        for key, b in self.buttons.items():
            active = key == current
            b.color = theme.GREEN_DARK if active else theme.TEXT_MUTED
            b.bold = active
            b.background_color = theme.GREEN_PALE if active else theme.WHITE


class PlantCareApp(App):
    title = "PlantCare AI"

    def __init__(self, ctx=None, **kw):
        super().__init__(**kw)
        self._ctx = ctx

    def build(self):
        if platform not in ("android", "ios"):
            Window.size = (dp(360), dp(720))
        Window.clearcolor = theme.CREAM
        self.ctx = self._ctx or AppContext(self.user_data_dir)

        self.sm = ScreenManager(transition=SlideTransition(duration=0.15))
        self.sm.scan_mode = None
        self.home = HomeScreen(ctx=self.ctx, open_record=self.open_record)
        self.scan = ScanScreen(ctx=self.ctx)
        self.scan.analyze_handler = self.start_analysis
        self.result = ResultScreen(self.ctx, goto=self.goto)
        self.disease = DiseaseDetailScreen(self.ctx, goto=self.goto)
        for s in (self.home, self.scan, self.result,
                  LearnScreen(self.ctx, open_disease=self.open_disease),
                  HistoryScreen(self.ctx, open_record=self.open_record),
                  SettingsScreen(self.ctx), AboutScreen(self.ctx), self.disease):
            self.sm.add_widget(s)

        root = BoxLayout(orientation="vertical")
        root.add_widget(self.sm)
        root.add_widget(NavBar(self.sm))
        Window.bind(on_keyboard=self._on_key)
        return root

    # -- navigation ------------------------------------------------------
    def goto(self, name):
        self.sm.current = name

    def open_record(self, record, back_to="home"):
        self.result.show(record, back_target=back_to)
        self.sm.current = "result"

    def open_disease(self, info):
        self.disease.show(info)
        self.sm.current = "disease"

    def _on_key(self, _w, key, *_):
        if key == 27:  # Android back / Esc
            cur = self.sm.current
            target = self.result.back_target if cur == "result" else BACK_MAP.get(cur)
            if target:
                self.sm.current = target
                return True
        return False

    # -- analysis (runs the model off the UI thread) -------------------------
    def start_analysis(self, image_path):
        if self.ctx.predictor is None:
            self.scan._set_error(self.ctx.model_error or "The disease model is not available.")
            return
        self.scan.set_busy(True)

        def work():
            try:
                rec, err = self.ctx.analyze(image_path), None
            except (ImageError, ModelError) as exc:
                rec, err = None, str(exc)
            except Exception as exc:  # database or unexpected failure: never crash the app
                rec, err = None, f"Something went wrong while analysing: {exc}"
            Clock.schedule_once(lambda _dt: self._analysis_done(rec, err), 0)
        threading.Thread(target=work, daemon=True).start()

    def _analysis_done(self, record, error):
        self.scan.set_busy(False)
        if error:
            self.scan._set_error(error + "  You can try again or choose a different photo.")
            return
        keep = record["saved"]
        if keep:
            self.scan.release_image()   # history owns the image file now
        else:
            self.scan.image_path = None  # result screen cleans up the temporary file on leave
            self.scan.clear_image()
        self.open_record(record, "scan")


if __name__ == "__main__":
    PlantCareApp().run()
