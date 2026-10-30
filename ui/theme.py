"""Central colour, spacing and font constants so the whole app looks consistent."""
from kivy.utils import get_color_from_hex as hex_

GREEN_DARK = hex_("#1B5E20")
GREEN = hex_("#2E7D32")
GREEN_LIGHT = hex_("#A5D6A7")
GREEN_PALE = hex_("#E8F5E9")
CREAM = hex_("#F7F5EC")
WHITE = hex_("#FFFFFF")
BROWN = hex_("#6D4C41")
TEXT = hex_("#1F2A1F")
TEXT_MUTED = hex_("#5F6B5F")
WARNING = hex_("#F9A825")
ERROR = hex_("#C62828")

# Spacing / sizing in density-independent pixels (use with kivy.metrics.dp)
PAD = 16
GAP = 12
RADIUS = 16
BUTTON_HEIGHT = 52
