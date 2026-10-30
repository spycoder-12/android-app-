"""Common base: cream background + a scrolling vertical column."""
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView

from ui import theme


class BaseScreen(Screen):
    def __init__(self, name, ctx=None, **kw):
        super().__init__(name=name, **kw)
        self.ctx = ctx
        with self.canvas.before:
            Color(*theme.CREAM)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._sync, size=self._sync)
        self.scroll = ScrollView(do_scroll_x=False, bar_width=dp(3))
        self.col = BoxLayout(orientation="vertical", size_hint_y=None, padding=dp(theme.PAD),
                             spacing=dp(theme.GAP))
        self.col.bind(minimum_height=self.col.setter("height"))
        self.scroll.add_widget(self.col)
        self.add_widget(self.scroll)

    def _sync(self, *_):
        self._bg.pos, self._bg.size = self.pos, self.size
