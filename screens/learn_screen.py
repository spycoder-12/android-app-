"""Searchable plant-disease information, photo guide and care tips; plus a detail screen."""
from kivy.metrics import dp, sp
from kivy.uix.textinput import TextInput

from screens.base import BaseScreen
from ui import theme
from ui.info_view import add_info_sections
from ui.rows import SimpleRow
from ui.widgets import Card, SecondaryButton, SectionTitle, WrapLabel


class LearnScreen(BaseScreen):
    def __init__(self, ctx, open_disease, **kw):
        super().__init__("learn", ctx=ctx, **kw)
        self.open_disease = open_disease
        self.col.add_widget(SectionTitle(text="Learn"))
        self.search = TextInput(hint_text="Search diseases or symptoms (e.g. blight, curl)", multiline=False,
                                size_hint_y=None, height=dp(46), font_size=sp(15), padding=[dp(12), dp(12)])
        self.search.bind(text=lambda *_: self.refresh())
        self.col.add_widget(self.search)
        self.results_box = self.col  # results appended after the search field
        self._dynamic = []
        self.refresh()

    def refresh(self):
        for w in self._dynamic:
            self.col.remove_widget(w)
        self._dynamic = []
        add = lambda w: (self.col.add_widget(w), self._dynamic.append(w))
        q = self.search.text
        items = self.ctx.info.search(q)
        add(SectionTitle(text="Tomato conditions" if not q.strip() else f"Results ({len(items)})", font_size=sp(16)))
        if not items:
            add(WrapLabel(text="Nothing matches that search. Try a disease name or a symptom such as 'spots'.",
                          color=theme.TEXT_MUTED))
        for i in items:
            add(SimpleRow(i["name"], i["category"], on_open=lambda it=i: self.open_disease(it)))
        if not q.strip():
            add(SectionTitle(text="How to photograph a leaf", font_size=sp(16)))
            c = Card(size_hint_y=None); c.bind(minimum_height=c.setter("height"))
            c.add_widget(WrapLabel(text="\n".join(f"{n}. {t}" for n, t in enumerate(self.ctx.info.photo_guide(), 1)),
                                   font_size=sp(14)))
            add(c)
            add(SectionTitle(text="Plant-care tips", font_size=sp(16)))
            for t in self.ctx.info.care_tips():
                c = Card(size_hint_y=None); c.bind(minimum_height=c.setter("height"))
                c.add_widget(WrapLabel(text=t["title"], bold=True, color=theme.GREEN_DARK, font_size=sp(14)))
                c.add_widget(WrapLabel(text=t["body"], font_size=sp(13)))
                add(c)


class DiseaseDetailScreen(BaseScreen):
    def __init__(self, ctx, goto, **kw):
        super().__init__("disease", ctx=ctx, **kw)
        self.goto = goto

    def show(self, info):
        self.col.clear_widgets()
        back = SecondaryButton(text="Back to Learn")
        back.bind(on_release=lambda *_: self.goto("learn"))
        self.col.add_widget(back)
        add_info_sections(self.col, info, self.ctx.info.disclaimer)
        self.scroll.scroll_y = 1
