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


    
    def train(self, config: str = "data.yaml") -> YOLO:

        if not Path(config).exists():
            raise RuntimeError(f'No file named {config} found.')
        
        t_args = {
            'data': config,
            'epochs': self.epochs,
        }
        self.base_model.train(**t_args)
        
