"""Modal dialogs: confirm() and info()."""
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup

from ui import theme
from ui.widgets import PrimaryButton, SecondaryButton, WrapLabel


def _popup(title, message, buttons, height=240):
    box = BoxLayout(orientation="vertical", padding=dp(14), spacing=dp(12))
    box.add_widget(WrapLabel(text=message, font_size=sp(15), size_hint_y=1))
    row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(10))
    for b in buttons:
        row.add_widget(b)
    box.add_widget(row)
    p = Popup(title=title, content=box, size_hint=(0.9, None), height=dp(height), auto_dismiss=False)
    return p


def confirm(title, message, on_yes, yes_text="Delete", no_text="Cancel"):
    no, yes = SecondaryButton(text=no_text), PrimaryButton(text=yes_text, fill=theme.ERROR)
    p = _popup(title, message, [no, yes])
    no.bind(on_release=lambda *_: p.dismiss())
    yes.bind(on_release=lambda *_: (p.dismiss(), on_yes()))
    p.open()
    return p


def info(title, message, ok_text="OK"):
    ok = PrimaryButton(text=ok_text)
    p = _popup(title, message, [ok], height=220)
    ok.bind(on_release=lambda *_: p.dismiss())
    p.open()
    return p
