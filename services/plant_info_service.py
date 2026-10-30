"""Loads disease/care information from JSON (kept separate from inference code)."""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class PlantInfoError(Exception):
    pass


class PlantInfoService:
    def __init__(self, data_dir=DATA_DIR):
        try:
            with open(os.path.join(data_dir, "diseases.json"), encoding="utf-8") as fh:
                self._d = json.load(fh)
            with open(os.path.join(data_dir, "care_tips.json"), encoding="utf-8") as fh:
                self._care = json.load(fh)
        except (OSError, ValueError) as exc:
            raise PlantInfoError(f"Disease information could not be loaded: {exc}") from exc
        self.disclaimer = self._d.get("disclaimer", "")

    def plants(self):
        return {k: v["display_name"] for k, v in self._d["plants"].items()}

    def get(self, plant, key):
        """Returns the info dict (with 'key' and 'plant') or None if unknown."""
        info = self._d["plants"].get(plant, {}).get("diseases", {}).get(key)
        return dict(info, key=key, plant=plant) if info else None

    def all(self, plant=None):
        out = []
        for p, pd in self._d["plants"].items():
            if plant and p != plant:
                continue
            for k, v in pd["diseases"].items():
                out.append(dict(v, key=k, plant=p, plant_name=pd["display_name"]))
        return sorted(out, key=lambda x: x["name"].lower())

    def search(self, query):
        q = (query or "").strip().lower()
        items = self.all()
        if not q:
            return items
        def hay(i):
            return " ".join([i["name"], i["category"], i["summary"], i["plant_name"],
                             *i["symptoms"]]).lower()
        return [i for i in items if all(t in hay(i) for t in q.split())]

    def care_tips(self):
        return self._care.get("tips", [])

    def photo_guide(self):
        return self._care.get("photo_guide", [])
