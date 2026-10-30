"""About: what the app is, how it works, real evaluation numbers, data credits, disclaimer."""
from kivy.metrics import sp

from screens.base import BaseScreen
from ui import theme
from ui.widgets import Card, SectionTitle, WrapLabel

APP_VERSION = "1.0.0"


class AboutScreen(BaseScreen):
    def __init__(self, ctx, **kw):
        super().__init__("about", ctx=ctx, **kw)

    def on_pre_enter(self, *_):
        col = self.col; col.clear_widgets()
        col.add_widget(SectionTitle(text="About PlantCare AI"))

        def card(title, body):
            c = Card(size_hint_y=None); c.bind(minimum_height=c.setter("height"))
            c.add_widget(WrapLabel(text=title, bold=True, color=theme.GREEN_DARK, font_size=sp(15)))
            c.add_widget(WrapLabel(text=body, font_size=sp(13)))
            col.add_widget(c)

        card("PlantCare AI", f"Version {APP_VERSION}\nAn on-device helper that suggests which common tomato leaf "
                             "condition a photo most resembles, with general care information.")
        card("How it works", "Your photo is resized and passed through a small neural network that was trained "
                             "on labelled leaf images. It outputs a score for each condition it knows. "
                             "The top score is shown, not a certainty.")
        p = self.ctx.predictor
        if p:
            ev = p.cfg.get("evaluation", {})
            stress = ev.get("stress_test_accuracy_simulated", {})
            card("Model evaluation (held-out test images)",
                 f"Test images: {ev.get('n_test')}\nAccuracy: {ev.get('accuracy', 0):.1%}   Macro F1: {ev.get('macro_f1', 0):.3f}\n"
                 "Simulated degradation (darker/brighter/blur/crop/noise) accuracy: "
                 + ", ".join(f"{k} {v:.0%}" for k, v in stress.items())
                 + "\n\nImportant: " + " ".join(ev.get("caveats", [])) +
                 " Accuracy on real photos taken in fields or gardens is expected to be lower.")
        else:
            card("Model", self.ctx.model_error or "Model not available.")
        card("Data credit", "Trained on the PlantVillage dataset (Hughes & Salathe; Mohanty, Hughes & Salathe, "
                            "2016, Frontiers in Plant Science).")
        card("Disclaimer", self.ctx.info.disclaimer)
        self.scroll.scroll_y = 1
