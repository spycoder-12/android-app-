"""Reusable Kivy widgets: Card, PrimaryButton, SecondaryButton, SectionTitle, Logo."""
from kivy.graphics import Color, Ellipse, Line, RoundedRectangle, Rotate, PushMatrix, PopMatrix
from kivy.metrics import dp, sp
from kivy.properties import ListProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget

from ui import theme


class Card(BoxLayout):
    """White rounded container with a subtle border."""
    bg_color = ListProperty(theme.WHITE)
    radius = NumericProperty(dp(theme.RADIUS))

    def __init__(self, **kwargs):
        kwargs.setdefault("orientation", "vertical")
        kwargs.setdefault("padding", dp(theme.PAD))
        kwargs.setdefault("spacing", dp(8))
        super().__init__(**kwargs)
        with self.canvas.before:
            self._c = Color(*self.bg_color)
            self._bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[self.radius])
            Color(*theme.GREEN_LIGHT)
            self._border = Line(rounded_rectangle=(*self.pos, *self.size, self.radius), width=1)
        self.bind(pos=self._redraw, size=self._redraw, bg_color=self._redraw)

    def _redraw(self, *_):
        self._c.rgba = self.bg_color
        self._bg.pos, self._bg.size = self.pos, self.size
        self._border.rounded_rectangle = (*self.pos, *self.size, self.radius)


class RoundedButton(Button):
    """Flat rounded button. Subclasses pick colours."""
    fill = ListProperty(theme.GREEN)
    text_color = ListProperty(theme.WHITE)
    outline = ListProperty([0, 0, 0, 0])

    def __init__(self, **kwargs):
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", dp(theme.BUTTON_HEIGHT))
        kwargs.setdefault("font_size", sp(16))
        kwargs.setdefault("bold", True)
        kwargs["background_normal"] = ""
        kwargs["background_down"] = ""
        kwargs["background_color"] = (0, 0, 0, 0)
        super().__init__(**kwargs)
        self.color = self.text_color
        with self.canvas.before:
            self._c = Color(*self.fill)
            self._bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(14)])
            self._oc = Color(*self.outline)
            self._ol = Line(rounded_rectangle=(*self.pos, *self.size, dp(14)), width=1.4)
        self.bind(pos=self._redraw, size=self._redraw, state=self._redraw)

    def _redraw(self, *_):
        pressed = self.state == "down"
        r, g, b, a = self.fill
        self._c.rgba = (r * 0.85, g * 0.85, b * 0.85, a) if pressed else (r, g, b, a)
        self._oc.rgba = self.outline
        self._bg.pos, self._bg.size = self.pos, self.size
        self._ol.rounded_rectangle = (*self.pos, *self.size, dp(14))


class PrimaryButton(RoundedButton):
    pass


class SecondaryButton(RoundedButton):
    def __init__(self, **kwargs):
        kwargs.setdefault("fill", theme.WHITE)
        kwargs.setdefault("text_color", theme.GREEN_DARK)
        kwargs.setdefault("outline", theme.GREEN)
        super().__init__(**kwargs)


class WrapLabel(Label):
    """Label that wraps to its width and grows in height to fit its text."""
    def __init__(self, **kwargs):
        kwargs.setdefault("color", theme.TEXT)
        kwargs.setdefault("halign", "left")
        kwargs.setdefault("valign", "top")
        kwargs.setdefault("size_hint_y", None)
        super().__init__(**kwargs)
        self.bind(width=self._wrap, texture_size=self._fit)
        self.texture_update()
        self._fit()          # empty labels must start at height 0, not Kivy's default 100

    def _wrap(self, *_):
        self.text_size = (self.width, None)

    def _fit(self, *_):
        self.height = self.texture_size[1]


class SectionTitle(WrapLabel):
    def __init__(self, **kwargs):
        kwargs.setdefault("font_size", sp(18))
        kwargs.setdefault("bold", True)
        kwargs.setdefault("color", theme.GREEN_DARK)
        super().__init__(**kwargs)


class Logo(Widget):
    """Simple vector leaf logo drawn on the canvas (no image asset needed)."""
    def __init__(self, **kwargs):
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("size", (dp(72), dp(72)))
        super().__init__(**kwargs)
        self.bind(pos=self._draw, size=self._draw)
        self._draw()

    def _draw(self, *_):
        self.canvas.clear()
        x, y = self.pos
        w, h = self.size
        with self.canvas:
            Color(*theme.GREEN)
            Ellipse(pos=(x, y), size=(w, h))
            # leaf body: rotated white ellipse
            PushMatrix()
            Rotate(angle=40, origin=(x + w / 2, y + h / 2))
            Color(*theme.WHITE)
            Ellipse(pos=(x + w * 0.30, y + h * 0.16), size=(w * 0.40, h * 0.68))
            Color(*theme.GREEN)
            Line(points=[x + w / 2, y + h * 0.20, x + w / 2, y + h * 0.80], width=dp(1.4))
            PopMatrix()
