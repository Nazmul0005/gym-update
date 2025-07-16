from fastapi import FastAPI
from fastapi import status

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from com.mhire.app.services.ai_coach.ai_coach_router import router as ai_coach_router
from com.mhire.app.services.food_scanner.food_scanner_router import router as food_scanner_router
from com.mhire.app.services.meal_planner.meal_planner_router import router as meal_planner_router
from com.mhire.app.services.workout_planner.workout_planner_router import router as workout_planner_router

app = FastAPI(
    title="Gym Coach API",
    description="AI-powered Gym and Health coaching application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ai_coach_router)
app.include_router(food_scanner_router)
app.include_router(meal_planner_router)
app.include_router(workout_planner_router)

@app.get("/", status_code=status.HTTP_200_OK, response_class=PlainTextResponse)
async def health_check():
    return "Server is running and healthy"