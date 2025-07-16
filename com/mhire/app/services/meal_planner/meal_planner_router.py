import logging
from fastapi import APIRouter, HTTPException
from com.mhire.app.services.meal_planner.meal_planner import MealPlanner
from com.mhire.app.services.meal_planner.meal_planner_schema import UserProfile, DailyMealPlan
from typing import Optional
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Meal Planner"]
)

meal_planner = MealPlanner()

class MealPlanResponse(BaseModel):
    success: bool = True
    meal_plan: Optional[DailyMealPlan] = None
    error: Optional[str] = None

@router.post("/meal-planner", response_model=MealPlanResponse)
async def generate_meal_plan(profile: UserProfile):
    """
    Generate a customized daily meal plan based on user profile
    """
    try:
        meal_plan = await meal_planner.generate_meal_plan(profile)
        return MealPlanResponse(
            success=True,
            meal_plan=meal_plan
        )
    except Exception as e:
        logger.error(f"Error in generate meal plan endpoint: {str(e)}")
        return MealPlanResponse(
            success=False,
            error=str(e)
        )