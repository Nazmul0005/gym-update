from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from datetime import date
from pydantic import BaseModel, Field, computed_field

class PrimaryGoal(str, Enum):
    BUILD_MUSCLE = "Build Muscle"
    LOSE_WEIGHT = "Lose Weight"
    EAT_HEALTHIER = "Eat Healthier"

class EatingStyle(str, Enum):
    VEGAN = "Vegan"
    KETO = "Keto"
    PALEO = "Paleo"
    VEGETARIAN = "Vegetarian"
    BALANCED = "Balanced"
    NONE = "None"

class ConsumptionFrequencyCaffine(str, Enum):
    NONE = "None"
    OCCASIONALLY = "Occasionally"
    DAILY = "Daily"

class ConsumptionFrequencySugar(str, Enum):
    NONE = "None"
    OCCASIONALLY = "Occasionally"
    CRAVEIT = "Crave it"
    DAILY = "Daily"

class WorkoutType(str, Enum):
    HOME = "Home"
    GYM = "Gym"


class WorkoutFrequency(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7




class UserProfileRequest(BaseModel):
    primary_goal: PrimaryGoal
    weight_kg: float
    height_cm: float
    is_meat_eater: bool
    is_lactose_intolerant: bool
    allergies: List[str]
    eating_style: EatingStyle
    caffeine_consumption: ConsumptionFrequencyCaffine
    sugar_consumption: ConsumptionFrequencySugar
    workout_type: WorkoutType
    workout_frequency: WorkoutFrequency
    date_of_birth: str = Field(..., description="Date of birth in YYYY/MM/DD format")
    
    @computed_field
    @property
    def age(self) -> int:
        """Calculate age from date of birth"""
        from datetime import datetime
        try:
            # Parse YYYY/MM/DD format
            birth_date = datetime.strptime(self.date_of_birth, "%Y/%m/%d").date()
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        except ValueError:
            return 0  # or handle error as needed
    

# Workout specific response models
class Exercise(BaseModel):
    name: str
    sets: int
    reps: str
    rest: str
    instructions: str
    video_url: Optional[str] = ""
    duration: Optional[int] = None  # Duration in seconds


class WorkoutSegment(BaseModel):
    motto: str
    exercises: List[Exercise]
    mark_complete: bool = False
    duration: int  # Total duration in seconds

class DailyWorkout(BaseModel):
    day: str
    focus: str
    warm_up: WorkoutSegment
    main_routine: WorkoutSegment
    cool_down: WorkoutSegment

class WorkoutResponse(BaseModel):
    success: bool = True
    workout_plan: List[DailyWorkout] = []  # Changed to required field with default empty list
    error: Optional[str] = None