"""Home screen: branding, primary actions, recent history (empty state for now), care tips."""
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen
from kivy.graphics import Color, Rectangle

from ui import theme
from ui.rows import HistoryRow
from ui.widgets import (Card, Logo, PrimaryButton, SecondaryButton,
                        SectionTitle, WrapLabel)

CARE_TIPS = [
    ("Water at the base", "Wet leaves stay damp longer, which favours many fungal and bacterial diseases."),
    ("Give plants space", "Good airflow helps foliage dry quickly after rain or watering."),
    ("Remove infected debris", "Fallen diseased leaves can carry pathogens to the next season."),
    ("Rotate crops", "Avoid planting the same crop family in the same bed year after year."),
]


class HomeScreen(Screen):
    def __init__(self, ctx=None, open_record=None, **kwargs):
        super().__init__(name="home", **kwargs)
        self.ctx, self.open_record = ctx, open_record
        with self.canvas.before:
            Color(*theme.CREAM)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._sync_bg, size=self._sync_bg)

        scroll = ScrollView(do_scroll_x=False, bar_width=dp(3))
        self.content = BoxLayout(orientation="vertical", size_hint_y=None,
                                 padding=dp(theme.PAD), spacing=dp(theme.GAP))
        self.content.bind(minimum_height=self.content.setter("height"))
        scroll.add_widget(self.content)
        self.add_widget(scroll)

        self._build_header()
        self._build_actions()
        self._build_recent()
        self._build_tips()
        self._build_disclaimer()

    # -- layout helpers -------------------------------------------------
    def _sync_bg(self, *_):
        self._bg.pos, self._bg.size = self.pos, self.size

    def _build_header(self):
        row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(80),
                        spacing=dp(14))
        row.add_widget(Logo())
        titles = BoxLayout(orientation="vertical")
        titles.add_widget(WrapLabel(text="PlantCare AI", font_size=sp(26), bold=True,
                                    color=theme.GREEN_DARK, valign="bottom"))
        titles.add_widget(WrapLabel(text="Leaf disease detection", font_size=sp(14),
                                    color=theme.TEXT_MUTED))
        row.add_widget(titles)
        self.content.add_widget(row)

    def _build_actions(self):
        card = Card(size_hint_y=None, bg_color=theme.GREEN_PALE)
        card.add_widget(WrapLabel(text="Welcome!", font_size=sp(20), bold=True,
                                  color=theme.GREEN_DARK))
        card.add_widget(WrapLabel(
            text="Take or choose a clear photo of a tomato leaf and get an AI-based "
                 "suggestion about its health, plus care advice.",
            font_size=sp(14), color=theme.TEXT))
        self.btn_start = PrimaryButton(text="Start Diagnosis")
        self.btn_upload = SecondaryButton(text="Upload Leaf Image")
        self.btn_capture = SecondaryButton(text="Capture Photo")
        for b in (self.btn_start, self.btn_upload, self.btn_capture):
            card.add_widget(b)
        self.btn_start.bind(on_release=lambda *_: self._go_scan("start"))
        self.btn_upload.bind(on_release=lambda *_: self._go_scan("upload"))
        self.btn_capture.bind(on_release=lambda *_: self._go_scan("capture"))
        card.bind(minimum_height=card.setter("height"))
        self.content.add_widget(card)

    def _build_recent(self):
        self.content.add_widget(SectionTitle(text="Recent detections"))
        self.recent_card = Card(size_hint_y=None, padding=dp(8))
        self.recent_card.bind(minimum_height=self.recent_card.setter("height"))
        self.set_recent([])
        self.content.add_widget(self.recent_card)

    def set_recent(self, records):
        """Show recent scans. `records` is a list of dicts; empty list shows the empty state.
        (History rows are wired to SQLite in Phase 6.)"""
        self.recent_card.clear_widgets()
        if not records:
            self.recent_card.add_widget(WrapLabel(
                text="No scans yet. Your analysed leaves will appear here, "
                     "stored only on this device.",
                font_size=sp(14), color=theme.TEXT_MUTED))
            return
        for r in records:
            self.recent_card.add_widget(HistoryRow(r, on_open=lambda rec: self.open_record(rec, "home")))

    def _build_tips(self):
        self.content.add_widget(SectionTitle(text="Plant-care basics"))
        card = Card(size_hint_y=None, spacing=dp(10))
        card.bind(minimum_height=card.setter("height"))
        for title, body in CARE_TIPS:
            card.add_widget(WrapLabel(text=title, bold=True, font_size=sp(15),
                                      color=theme.GREEN_DARK))
            card.add_widget(WrapLabel(text=body, font_size=sp(13), color=theme.TEXT))
        self.content.add_widget(card)

    def _build_disclaimer(self):
        self.content.add_widget(WrapLabel(
            text="Results are model estimates, not a confirmed diagnosis. "
                 "Consult a local agricultural expert before treating crops.",
            font_size=sp(12), color=theme.TEXT_MUTED, italic=True))

    def on_pre_enter(self, *_):
        if not self.ctx:
            return
        try:
            self.set_recent(self.ctx.db.list(limit=3))
        except Exception as exc:  # DatabaseError: show message, keep app usable
            self.recent_card.clear_widgets()
            self.recent_card.add_widget(WrapLabel(text=str(exc), color=theme.ERROR, font_size=sp(13)))
        self._model_banner()

    def _model_banner(self):
        if self.ctx.model_error and not getattr(self, "_banner_added", False):
            self._banner_added = True
            c = Card(size_hint_y=None, bg_color=(1.0, 0.93, 0.93, 1))
            c.bind(minimum_height=c.setter("height"))
            c.add_widget(WrapLabel(text="Analysis unavailable: " + self.ctx.model_error, color=theme.ERROR,
                                   font_size=sp(13)))
            self.content.add_widget(c, index=len(self.content.children) - 2)

    def _go_scan(self, mode):
        self.manager.scan_mode = mode  # read by the scan screen in Phase 3
        self.manager.current = "scan"
