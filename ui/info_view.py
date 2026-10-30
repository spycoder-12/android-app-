"""Shared renderer for a disease-information dict (used by Result and Learn screens)."""
from kivy.metrics import dp, sp

from ui import theme
from ui.widgets import Card, SectionTitle, WrapLabel


def _bullets(items):
    return "\n".join(f"\u2022 {i}" for i in items)


def add_info_sections(col, info, disclaimer=""):
    """Append About / Symptoms / Prevention / Care / Expert-advice cards to `col`."""
    if not info:
        col.add_widget(WrapLabel(text="No detailed information is available for this result yet.",
                                 color=theme.TEXT_MUTED))
        return
    col.add_widget(SectionTitle(text=info["name"]))
    col.add_widget(WrapLabel(text=info["category"], font_size=sp(13), color=theme.BROWN, bold=True))
    for title, body in (("About", info["summary"]),
                        ("Common visible symptoms", _bullets(info["symptoms"])),
                        ("Prevention", _bullets(info["prevention"])),
                        ("Plant-care practices", _bullets(info["care"])),
                        ("When to ask an expert", info["expert_advice"])):
        card = Card(size_hint_y=None)
        card.bind(minimum_height=card.setter("height"))
        card.add_widget(WrapLabel(text=title, bold=True, font_size=sp(15), color=theme.GREEN_DARK))
        card.add_widget(WrapLabel(text=body, font_size=sp(14)))
        col.add_widget(card)
    if disclaimer:
        col.add_widget(WrapLabel(text=disclaimer, font_size=sp(11), italic=True, color=theme.TEXT_MUTED))
