"""Shows one analysis result (fresh or from history) with disease information."""
import os

from kivy.metrics import dp, sp
from kivy.uix.image import Image as KivyImage
from kivy.uix.progressbar import ProgressBar

from screens.base import BaseScreen
from ui import dialogs, theme
from ui.info_view import add_info_sections
from ui.widgets import Card, PrimaryButton, SecondaryButton, SectionTitle, WrapLabel

GENERIC_INCONCLUSIVE = ("The model was not confident about this image. Retake a clearer, closer photo "
                        "of a single leaf, or ask an expert.")


class ResultScreen(BaseScreen):
    def __init__(self, ctx, goto, **kw):
        super().__init__("result", ctx=ctx, **kw)
        self.goto = goto               # goto(screen_name)
        self.record = None
        self.back_target = "home"

    def show(self, record, back_target="home"):
        self.record, self.back_target = record, back_target
        col = self.col
        col.clear_widgets()
        col.add_widget(SectionTitle(text="Result"))

        img_card = Card(size_hint_y=None, height=dp(220))
        if record.get("image_exists") and os.path.isfile(record["image_path"]):
            img_card.add_widget(KivyImage(source=record["image_path"], allow_stretch=True, keep_ratio=True))
        else:
            img_card.add_widget(WrapLabel(text="The original image is no longer available.",
                                          color=theme.TEXT_MUTED, halign="center", valign="middle",
                                          size_hint_y=1))
        col.add_widget(img_card)

        inconclusive = record["status"] == "inconclusive"
        banner = Card(size_hint_y=None, bg_color=(1.0, 0.97, 0.88, 1) if inconclusive else theme.GREEN_PALE)
        banner.bind(minimum_height=banner.setter("height"))
        if inconclusive:
            banner.add_widget(WrapLabel(text="Inconclusive result", bold=True, font_size=sp(13), color=theme.BROWN))
        banner.add_widget(WrapLabel(text=("Best guess: " if inconclusive else "Most likely: ") + record["predicted_label"],
                                    bold=True, font_size=sp(21), color=theme.GREEN_DARK))
        banner.add_widget(WrapLabel(text=f"Model score: {record['confidence']:.0%}  (a model estimate, "
                                         "not a guarantee that this is correct)", font_size=sp(13)))
        bar = ProgressBar(max=100, value=round(record["confidence"] * 100), size_hint_y=None, height=dp(10))
        banner.add_widget(bar)
        col.add_widget(banner)

        warnings = record.get("warnings") or ([GENERIC_INCONCLUSIVE] if inconclusive else [])
        if warnings:
            wc = Card(size_hint_y=None, bg_color=(1.0, 0.97, 0.88, 1))
            wc.bind(minimum_height=wc.setter("height"))
            wc.add_widget(WrapLabel(text="Please note", bold=True, font_size=sp(14), color=theme.BROWN))
            wc.add_widget(WrapLabel(text="\n".join(f"\u2022 {w}" for w in warnings), font_size=sp(13)))
            col.add_widget(wc)

        if record.get("top_k"):
            tc = Card(size_hint_y=None)
            tc.bind(minimum_height=tc.setter("height"))
            tc.add_widget(WrapLabel(text="Other possibilities (model scores)", bold=True, font_size=sp(14),
                                    color=theme.GREEN_DARK))
            tc.add_widget(WrapLabel(text="\n".join(f"{t['label']}: {t['prob']:.0%}" for t in record["top_k"]),
                                    font_size=sp(13)))
            col.add_widget(tc)

        info = self.ctx.info.get(record["plant"], record["predicted_key"])
        add_info_sections(col, info, self.ctx.info.disclaimer)

        again = PrimaryButton(text="Scan another leaf")
        again.bind(on_release=lambda *_: self.goto("scan"))
        col.add_widget(again)
        if record.get("saved") or record.get("id"):
            d = SecondaryButton(text="Delete this record", text_color=theme.ERROR, outline=theme.ERROR)
            d.bind(on_release=lambda *_: dialogs.confirm(
                "Delete this record?", "This removes the result and its photo from this device.",
                self._delete))
            col.add_widget(d)
        self.scroll.scroll_y = 1

    def _delete(self):
        try:
            self.ctx.db.delete(self.record["id"])
        except Exception as exc:  # DatabaseError
            dialogs.info("Could not delete", str(exc))
            return
        self.record = None
        self.goto("history" if self.back_target == "history" else "home")

    def on_leave(self, *_):
        """Unsaved results keep a temporary image; remove it once the user leaves."""
        r = self.record
        if r and not r.get("id") and r.get("image_path") and os.path.isfile(r["image_path"]):
            try:
                os.remove(r["image_path"])
            except OSError:
                pass
