"""Desktop image picker: Kivy popup with a file chooser (used on Windows/Linux/macOS)."""
import os
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup

from ui.widgets import PrimaryButton, SecondaryButton

_EXTS = ("jpg", "jpeg", "png", "webp", "bmp")
FILTERS = [f"*.{e}" for e in _EXTS] + [f"*.{e.upper()}" for e in _EXTS]


def open_desktop_picker(on_pick):
    start = os.path.join(os.path.expanduser("~"), "Pictures")
    if not os.path.isdir(start):
        start = os.path.expanduser("~")
    chooser = FileChooserListView(path=start, filters=FILTERS)
    box = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(8))
    box.add_widget(chooser)
    row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
    cancel, ok = SecondaryButton(text="Cancel"), PrimaryButton(text="Select")
    row.add_widget(cancel)
    row.add_widget(ok)
    box.add_widget(row)
    popup = Popup(title="Choose a leaf image", content=box, size_hint=(0.95, 0.9))

    def select(*_):
        popup.dismiss()
        if chooser.selection:
            on_pick(chooser.selection[0])
    chooser.bind(on_submit=lambda _c, sel, _t: select())
    ok.bind(on_release=select)
    cancel.bind(on_release=lambda *_: popup.dismiss())
    popup.open()
    return popup
