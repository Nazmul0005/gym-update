from fastapi import APIRouter, HTTPException
from com.mhire.app.services.workout_planner.workout_planner import WorkoutPlanner
from com.mhire.app.services.workout_planner.workout_planner_schema import UserProfileRequest, WorkoutResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["Workout Planner"]
)

@router.post("/workout-planner", response_model=WorkoutResponse)
async def generate_workout_plan(request: UserProfileRequest):
    """
    Generate a personalized workout plan based on user parameters
    """
    try:
        planner = WorkoutPlanner()
        plan = await planner.generate_workout_plan(request)
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))