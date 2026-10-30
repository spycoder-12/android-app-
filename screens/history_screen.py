"""History: list, open, delete one, clear all (with confirmation)."""
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout

from database.db_manager import DatabaseError
from screens.base import BaseScreen
from ui import dialogs, theme
from ui.rows import HistoryRow
from ui.widgets import SecondaryButton, SectionTitle, WrapLabel


class HistoryScreen(BaseScreen):
    def __init__(self, ctx, open_record, **kw):
        super().__init__("history", ctx=ctx, **kw)
        self.open_record = open_record

    def on_pre_enter(self, *_):
        self.refresh()

    def refresh(self):
        col = self.col
        col.clear_widgets()
        head = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        head.add_widget(SectionTitle(text="History", size_hint_y=1, valign="middle"))
        try:
            records = self.ctx.db.list()
        except DatabaseError as exc:
            col.add_widget(head)
            col.add_widget(WrapLabel(text=str(exc), color=theme.ERROR, bold=True))
            return
        if records:
            clear = SecondaryButton(text="Clear all", size_hint=(None, None), size=(dp(100), dp(40)),
                                    font_size=sp(13), text_color=theme.ERROR, outline=theme.ERROR)
            clear.bind(on_release=lambda *_: dialogs.confirm(
                "Clear all history?", f"This permanently deletes {len(records)} saved result(s) and their "
                "photos from this device.", self._clear_all, yes_text="Clear all"))
            head.add_widget(clear)
        col.add_widget(head)
        if not records:
            col.add_widget(WrapLabel(
                text="No saved scans yet. Analysed leaves are stored only on this device and will appear here.",
                color=theme.TEXT_MUTED))
        for r in records:
            col.add_widget(HistoryRow(r, on_open=lambda rec: self.open_record(rec, "history"),
                                      on_delete=self._delete_one))
        self.scroll.scroll_y = 1

    def _delete_one(self, record):
        dialogs.confirm("Delete this record?", f"{record['predicted_label']} - removes the result and photo.",
                        lambda: self._do_delete(record["id"]))

    def _do_delete(self, rid):
        try:
            self.ctx.db.delete(rid)
        except DatabaseError as exc:
            dialogs.info("Could not delete", str(exc))
        self.refresh()

    def _clear_all(self):
        try:
            self.ctx.db.clear_all()
        except DatabaseError as exc:
            dialogs.info("Could not clear history", str(exc))
        self.refresh()
