from ultralytics import YOLO
import torch
import numpy as np
import queue
import threading
import asyncio

class Model:
    def __init__(self, model: YOLO, name: str, _id: str):
        self.model: YOLO = model
        self.name = name
        self._id = _id
        self.task_queue = queue.Queue()
        self.res_queue = queue.Queue()
        self.run = threading.Event()
        self.run.set()
        self._warm()
        self.i_t = threading.Thread(target=self._loop,daemon=True).start()

    def _warm(self):
        try:
            dr = np.zeros((320,320,3),dtype=np.unit8)
            print(f'warming {self.name}')
            return self.model.predict(dr,imgsz=320,verbose=False)
        except:
            return False

    
    async def predict(self, img: np.ndarray):
        self.task_queue.put(img)
        loop = asyncio.get_event_loop()
        f = loop.create_future()

        def get():
            try:
                succcess,result = self.res_queue.get_nowait()
                if succcess:
                    f.set_result(result)
                else:
                    f.set_exception(Exception(result))
            except queue.Empty:
                loop.call_later(0.02,get)
        loop.call_later(0.01,get)
        return await f
    
    def _loop(self):
        while self.run.is_set():
            try:
                img = self.task_queue.get(timeout=2.0)
                result = self.model.predict(img,imgsz=640,verbose=False)
                self.res_queue.put((True,result))
                self.task_queue.task_done()
            except queue.Empty:
                continue

    def benchmark(self):
        res = self.model.benchmark(imgsz=640)
        return res
    
    def switch_device(self, device: str):
        target_device = torch.device(device.lower())
        if device == "cuda" and not torch.cuda.is_available():
            return False
        self.model.to(target_device)
    
    @property
    def id(self):
        return self._id
    
    @property
    def classes(self) -> dict[int,str]:
        return self.model.names
    
    @property
    def conf_threshold(self):
        return self.model.overrides.get('conf',None)
    
    @property
    def type(self):
        return str(type(self.model).__name__)

    def __repr__(self) -> str:
        return f"Model(name={self.name}, id={self._id}, type={self.type})"
