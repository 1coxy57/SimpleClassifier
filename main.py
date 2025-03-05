import asyncio
from fastapi import FastAPI, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates  
from pydantic import BaseModel
from typing import List, Dict, Optional
from io import BytesIO
import time
import uuid
import cv2
import numpy as np
import base64
from models.loader import load_models 
import uvicorn
from fastapi.staticfiles import StaticFiles
from consts import *
from multiprocessing import Pool
from functools import lru_cache
from collections import deque



FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0,255,0)

def annotate(img: np.ndarray,detections: List[Dict]) -> str:
    for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_n, conf = det["class_name"], det["confidence"]
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(img, f"{class_n} ({conf:.2f})", (x1, y1 - 10), FONT, 0.5, GREEN, 2)
    success, i_enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY,75])
    return base64.b64encode(i_enc.tobytes()).decode('utf-8') if success else None

class App:
    def __init__(self):
        self.app = FastAPI(
            docs_url=None,
            openapi_url=None,
        )
        self.pool = Pool(processes=2)
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.templates = Jinja2Templates(directory="static/html") 
        self.setup_routes()
        self.base_models = {m.name: m for m in load_models()}
        self.temp = deque(maxlen=1000) # deques the most left item if the length of temp is >= 1000
        self.t: int = time.time()


    def setup_routes(self):
        @self.app.post("/api/classify", response_model=ClassificationResponse)
        async def classify_img(file: UploadFile, model: str = Form(...), rci: str = Form(...), tasks: BackgroundTasks = None):
            t = time.time()
            # .hex faster than str lol
            #abt 14.70% faster lol
            id_ = uuid.uuid4().hex
            try:
                contents = await file.read()
                img_array = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("img not readable")
                height, width, _ = img.shape
                file_type = file.filename.rpartition('.')[-1] or None
            except Exception as e:
                raise HTTPException(status_code=400,detail="json error")

            curr_model = self.base_models.get(model)
            if not curr_model:
                raise HTTPException(status_code=404, detail="Model not found")

            results = await curr_model.predict(img)
            detections = [
                {
                    "class_name": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": list(map(round,box.xyxy[0].tolist()))
                }
                for result in results
                for box in result.boxes
            ]
            
            def process_img(img,dets,result):
                bimg = self.pool.apply(annotate,(img.copy(),detections))
                if bimg:
                    self.bimg = bimg

            data = {
                "success": True,
                "result": {
                    "_id": id_,
                    "classified_image": None,
                    "item": detections,
                    "image": {
                        "size": [width,height],
                        "type": file_type,
                    },
                    "model": {
                        "name": model,
                        "confidence_threshold": curr_model.conf_threshold
                    },
                    "took": round(time.time() - t, 3)
                }
            }
            tasks.add_task(process_img,img,detections,data)
            data['result']['classified_image'] = self.bimg
            return data

        @self.app.get('/api/devices')
        async def ww(model: BenchmarkData):
            model = self.base_models.get(model.model)
            devices = model.devices
            return {"success": True, "devices": devices}

        @self.app.get('/api/classified',response_model=ModelsResponse)
        async def classified_imgs():
            return {"success": True, "models": self.temp}
        
        @lru_cache(maxsize=1)
        def get_active_models(self):
            models = [
                {"name": m.name, "_id": m.id, "type": m.type}
                for m in self.base_models.values()
            ]

        @self.app.get("/api/models", response_model=ModelsResponse)
        async def active_models():
            return {"success": True, "models": self.get_active_models()}

        @self.app.get("/api/uptime")
        async def uptime():
            return {"success": True, "uptime": round(time.time() - self.t,2)}

        @self.app.post("/api/benchmark", response_model=BenchmarkResponse)
        async def benchmark_model(model: BenchmarkData):
            # slow asf benchmark
            curr_model = self.base_models.get(model.model)
            if not curr_model:
                raise HTTPException(status_code=404, detail="Model not found")
            result = curr_model.benchmark()
            return {
                "success": True,
                "benchmark_results": {
                    "params": result.get("params", "N/A"),
                    "flops": result.get("flops", "N/A"),
                    "speed": result.get("latency", 0),
                    "fps": result.get("throughput", 0),
                }
            }


        @self.app.get("/")
        async def main(request: Request): 
            return self.templates.TemplateResponse(
                "main.html", 
                {"request": request}
            )

    def __call__(self, *args, **kwargs):
        uvicorn.run(
            self.app, 
            host="127.0.0.1", 
            port=8000, 
            *args, 
            **kwargs
        )

if __name__ == "__main__":
    app = App()
    app()
