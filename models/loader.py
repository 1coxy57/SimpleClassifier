from models.base_model import Model
from ultralytics import YOLO
import uuid
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
]

class NoBaseModels(Exception):
    pass

def load_models(models: list[str] = BASE_MODELS,path: str | Path = "./models/models") -> list[Model]:
    
    base_models: list[Model] = []
    weights = Path(path)

    for mod in models:
        dir = weights / mod
        try:
            if not dir.exists():
                logging.info(f"{m} does not exist in the current directory.")
                continue

            yolo_model = YOLO(str(dir))

            model_id = str(uuid.uuid4())
            model = Model(yolo_model,mod,model_id)
            base_models.append(model)

        except Exception as e:
            logging.info(f"Failed to load {m}: {str(e)}")
            raise RuntimeError("Failed loading")

    if not base_models:
        raise NoBaseModels(f"No base models loaded.")
    return base_models
