"""Tiny JSON-backed user settings."""
import json
import os

DEFAULTS = {"save_history": True, "show_quality_hints": True}


class Settings:
    def __init__(self, path):
        self.path = path
        self._v = dict(DEFAULTS)
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            self._v.update({k: loaded[k] for k in DEFAULTS if k in loaded})
        except (OSError, ValueError):
            pass   # first run or damaged file -> defaults

    def get(self, key):
        return self._v[key]

    def set(self, key, value):
        if key not in DEFAULTS:
            raise KeyError(key)
        self._v[key] = value
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self._v, fh)
        except OSError:
            pass
