from pydantic import BaseModel
from typing import List, Optional

class NutritionInfo(BaseModel):
    calories: float
    protein: float
    carbs: float
    fat: float

class FoodAnalysis(BaseModel):
    food_items: List[str]
    nutrition: NutritionInfo
    health_benefits: List[str]
    concerns: List[str]

class FoodScanResponse(BaseModel):
    success: bool
    analysis: Optional[FoodAnalysis] = None
    error: Optional[str] = None