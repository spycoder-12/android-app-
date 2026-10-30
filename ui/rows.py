"""List rows: HistoryRow (thumbnail + result + delete) and SimpleRow (tappable title/subtitle)."""
import os
from datetime import datetime

from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label

from ui import theme
from ui.widgets import Card, WrapLabel


class _Tappable(Card):
    """Card that fires on_open only for a real tap (not a scroll drag)."""
    def __init__(self, on_open=None, **kw):
        super().__init__(**kw)
        self.on_open = on_open
        self._down = None
        self.ignore_widgets = []

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._down = touch.pos
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        handled = super().on_touch_up(touch)
        if (self._down and self.collide_point(*touch.pos) and self.on_open
                and abs(touch.x - self._down[0]) < dp(10) and abs(touch.y - self._down[1]) < dp(10)
                and not any(w.collide_point(*touch.pos) for w in self.ignore_widgets)):
            self._down = None
            self.on_open()
            return True
        self._down = None
        return handled


def format_when(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return str(iso)


class HistoryRow(_Tappable):
    def __init__(self, record, on_open, on_delete=None, **kw):
        super().__init__(on_open=lambda: on_open(record), orientation="horizontal",
                         size_hint_y=None, height=dp(84), padding=dp(8), spacing=dp(10), **kw)
        self.record = record
        if record.get("image_exists"):
            self.add_widget(KivyImage(source=record["image_path"], size_hint=(None, 1), width=dp(64),
                                      allow_stretch=True, keep_ratio=True))
        else:
            self.add_widget(Label(text="No\nimage", size_hint=(None, 1), width=dp(64), font_size=sp(11),
                                  color=theme.TEXT_MUTED, halign="center"))
        mid = BoxLayout(orientation="vertical", spacing=dp(2))
        label = record["predicted_label"] + ("  (inconclusive)" if record["status"] == "inconclusive" else "")
        mid.add_widget(WrapLabel(text=label, bold=True, font_size=sp(15), color=theme.GREEN_DARK))
        mid.add_widget(WrapLabel(text=f"Model score {record['confidence']:.0%}",
                                 font_size=sp(12), color=theme.TEXT))
        mid.add_widget(WrapLabel(text=format_when(record["created_at"]), font_size=sp(11),
                                 color=theme.TEXT_MUTED))
        self.add_widget(mid)
        if on_delete:
            from ui.widgets import SecondaryButton
            btn = SecondaryButton(text="Delete", size_hint=(None, None), size=(dp(72), dp(40)),
                                  font_size=sp(12), text_color=theme.ERROR, outline=theme.ERROR,
                                  pos_hint={"center_y": 0.5})
            btn.bind(on_release=lambda *_: on_delete(record))
            holder = BoxLayout(size_hint=(None, 1), width=dp(72))
            holder.add_widget(btn)
            self.add_widget(holder)
            self.ignore_widgets = [holder]


class SimpleRow(_Tappable):
    def __init__(self, title, subtitle, on_open, **kw):
        super().__init__(on_open=on_open, size_hint_y=None, padding=dp(12), spacing=dp(2), **kw)
        self.add_widget(WrapLabel(text=title, bold=True, font_size=sp(16), color=theme.GREEN_DARK))
        self.add_widget(WrapLabel(text=subtitle, font_size=sp(12), color=theme.TEXT_MUTED))
        self.bind(minimum_height=self.setter("height"))
