from ultralytics import YOLO
import logging
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import os
import json
import time
import torch
from dataclasses import dataclass
import shutil


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class callback:
    ep_end: Optional[Callable[[int,Dict],None]] = None
    train_start: Optional[Callable[[int,Dict],None]] = None
    train_end: Optional[Callable[[int,Dict],None]] = None

class Trainer:
    def __init__(
        self,
        path: str = "yolov8.pt",
        config: Optional[Dict[str,Any]] = None,
        epochs: int = 100,
        save_dir: str | Path = "runs/train",
    ):
        self.epochs = epochs
        self.save_dir = Path(save_dir)

        self.callbacks: List[callback] = []

        self.base_model: YOLO = YOLO(path)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config or {}

        self.results = {}

    
    def train(self, config: str = "data.yaml") -> YOLO:
        t = time.time()

        if not Path(config).exists():
            raise RuntimeError(f'No file named {config} found.')
        
        t_args = {
            'data': config,
            'epochs': self.epochs,
            'device': self.device,
            'project': 'sad',
            'name': 'asdasd',
            **self.config,
        }
        results = self.base_model.train(**t_args)

        self.results = {
            'training_time': time.time() - t,
            'metrics': results.metrics if hasattr(results,'metrics') else {},
            'path': str(self.save_dir / "weights" / "best.pt"),
        }

        self._run_callbacks("ep_end",self.results)
        
    def add_callback(self,callback: callback):
        self.callbacks.append(callback)
        logging.info(f"Added new callback to trainer.")

    def _run_callbacks(self, hook: str = "",*args):
        for callback in self.callbacks:
            e = getattr(self.callbacks,hook,None)
            if e is not None:
                try:
                    e(*args)
                except:
                    logger.info(f"Failed to call hook")


    
