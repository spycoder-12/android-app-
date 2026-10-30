"""Settings: privacy toggles, history management, model information."""
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.switch import Switch

from database.db_manager import DatabaseError
from screens.base import BaseScreen
from ui import dialogs, theme
from ui.widgets import Card, SecondaryButton, SectionTitle, WrapLabel


class SettingsScreen(BaseScreen):
    def __init__(self, ctx, **kw):
        super().__init__("settings", ctx=ctx, **kw)

    def on_pre_enter(self, *_):
        self.build()

    def _toggle(self, key, title, desc):
        row = BoxLayout(size_hint_y=None, height=dp(64), spacing=dp(10))
        txt = BoxLayout(orientation="vertical")
        txt.add_widget(WrapLabel(text=title, bold=True, font_size=sp(15)))
        txt.add_widget(WrapLabel(text=desc, font_size=sp(12), color=theme.TEXT_MUTED))
        sw = Switch(active=self.ctx.settings.get(key), size_hint=(None, 1), width=dp(80))
        sw.bind(active=lambda _w, v: self.ctx.settings.set(key, v))
        row.add_widget(txt); row.add_widget(sw)
        return row

    def build(self):
        col = self.col; col.clear_widgets()
        col.add_widget(SectionTitle(text="Settings"))
        card = Card(size_hint_y=None); card.bind(minimum_height=card.setter("height"))
        card.add_widget(self._toggle("save_history", "Save results to history",
                                     "Keeps the photo and result on this device only."))
        card.add_widget(self._toggle("show_quality_hints", "Show photo-quality hints",
                                     "Warns about dark, blurry or low-resolution photos."))
        col.add_widget(card)

        pc = Card(size_hint_y=None); pc.bind(minimum_height=pc.setter("height"))
        pc.add_widget(WrapLabel(text="Privacy", bold=True, color=theme.GREEN_DARK, font_size=sp(15)))
        pc.add_widget(WrapLabel(text="Analysis runs on your phone. Photos and history stay on this device "
                                     "and are never uploaded.", font_size=sp(13)))
        try:
            n = self.ctx.db.count()
            pc.add_widget(WrapLabel(text=f"Saved results: {n}", font_size=sp(13), color=theme.TEXT_MUTED))
            clear = SecondaryButton(text="Clear all history", text_color=theme.ERROR, outline=theme.ERROR,
                                    disabled=(n == 0))
            clear.bind(on_release=lambda *_: dialogs.confirm(
                "Clear all history?", "This permanently deletes all saved results and photos from this device.",
                self._clear, yes_text="Clear all"))
            pc.add_widget(clear)
        except DatabaseError as exc:
            pc.add_widget(WrapLabel(text=str(exc), color=theme.ERROR))
        col.add_widget(pc)

        mc = Card(size_hint_y=None); mc.bind(minimum_height=mc.setter("height"))
        mc.add_widget(WrapLabel(text="Model", bold=True, color=theme.GREEN_DARK, font_size=sp(15)))
        p = self.ctx.predictor
        if p:
            lines = [f"Crop: {p.plant.title()}", f"Version: {p.version}", f"Engine: {getattr(p.engine, 'name', 'unknown')}",
                     f"Classes: {len(p.keys)}", f"Input: {p.size}x{p.size} RGB",
                     f"Inconclusive below {p.min_conf:.0%} score"]
            mc.add_widget(WrapLabel(text="\n".join(lines), font_size=sp(13)))
        else:
            mc.add_widget(WrapLabel(text=self.ctx.model_error or "Model not available.", color=theme.ERROR,
                                    font_size=sp(13)))
        col.add_widget(mc)

    def _clear(self):
        try:
            self.ctx.db.clear_all()
        except DatabaseError as exc:
            dialogs.info("Could not clear history", str(exc))
        self.build()
