from pydantic import BaseModel
from typing import List
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


class UserProfile(BaseModel):
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
    


class Meal(BaseModel):
    name: str
    description: str
    calories: float
    protein: float
    carbs: float
    fat: float
    rationale: str
    preparation_steps: List[str]

class DailyMealPlan(BaseModel):
    breakfast: Meal
    lunch: Meal
    snack: Meal
    dinner: Meal