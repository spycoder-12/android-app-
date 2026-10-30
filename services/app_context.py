"""Wires the services together; one instance per app run."""
import os

from database.db_manager import DBManager
from services import image_service
from services.plant_info_service import PlantInfoService
from services.prediction_service import MODEL_DIR, ModelError, PredictionService
from services.settings_service import Settings


class AppContext:
    def __init__(self, data_dir, model_dir=MODEL_DIR, info=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "user_images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.db = DBManager(os.path.join(data_dir, "history.db"))
        self.info = info or PlantInfoService()
        self.settings = Settings(os.path.join(data_dir, "settings.json"))
        self.predictor, self.model_error = None, None
        try:
            self.predictor = PredictionService(model_dir)
        except ModelError as exc:
            self.model_error = str(exc)

    def analyze(self, image_path):
        """Run the model and (if enabled) save to history. Returns a record dict for ResultScreen.
        Raises ModelError / ImageError with user-safe messages."""
        if self.predictor is None:
            raise ModelError(self.model_error or "The disease model is not available.")
        pred = self.predictor.predict(image_path)
        record = {"id": None, "image_path": image_path, "image_exists": os.path.isfile(image_path),
                  "plant": pred.plant, "predicted_key": pred.top_key, "predicted_label": pred.top_label,
                  "confidence": pred.confidence, "status": pred.status, "top_k": pred.top_k,
                  "warnings": pred.warnings, "model_version": pred.model_version, "created_at": "",
                  "saved": False}
        from datetime import datetime
        record["created_at"] = datetime.now().isoformat(timespec="seconds")
        if self.settings.get("save_history"):
            record["id"] = self.db.add_detection(
                image_path=image_path, plant=pred.plant, predicted_key=pred.top_key,
                predicted_label=pred.top_label, confidence=pred.confidence, status=pred.status,
                top_k=pred.top_k, model_version=pred.model_version)
            record["saved"] = True
        return record
