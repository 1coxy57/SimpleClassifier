from pydantic import BaseModel
from typing import Optional,List

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class ImageInfo(BaseModel):
    size: list[int,int]
    type: str

class ModelInfo(BaseModel):
    name: str
    confidence_threshold: Optional[float]

class ClassificationResponse(BaseModel):
    success: bool
    result: dict

class ModelsResponse(BaseModel):
    success: bool
    models: List[dict]

class BenchmarkResponse(BaseModel):
    success: bool
    benchmark_results: dict

class BenchmarkData(BaseModel):
    model: str
