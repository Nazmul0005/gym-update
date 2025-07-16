import logging
import re
import json
import time
from fastapi import HTTPException
from openai import OpenAI
from com.mhire.app.config.config import Config
from .meal_planner_schema import UserProfile, DailyMealPlan, Meal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MealPlanner:
    def __init__(self):
        try:
            config = Config()
            self.openai_client = OpenAI(api_key=config.openai_api_key)
            self.model = config.model_name2
        except Exception as e:
            logger.error(f"Error initializing MealPlanner: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Meal Planner: {str(e)}")

    def _create_meal_from_json(self, meal_json):
        """Create a Meal object from JSON data"""
        return Meal(
            name=meal_json["name"],
            description=meal_json["description"],
            calories=float(meal_json["calories"]),
            protein=float(meal_json["protein"]),
            carbs=float(meal_json["carbs"]),
            fat=float(meal_json["fat"]),
            rationale=meal_json["rationale"],
            preparation_steps=meal_json["preparation_steps"]
        )
    
    async def _get_ai_response(self, prompt: str, max_retries=3) -> str:
        """Get meal plan from OpenAI using the same method as workout_planner with retry mechanism"""
        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Attempt {retries + 1} to get AI response")
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a nutrition expert that provides meal plans in valid JSON format only. Your responses must be properly formatted JSON with no additional text or markdown formatting."}, 
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,  # Lower temperature for more consistent formatting
                    response_format={"type": "json_object"}  # Explicitly request JSON response
                )
                
                content = response.choices[0].message.content
                
                # Check if content is empty or whitespace
                if not content or content.isspace():
                    logger.warning("Received empty response from OpenAI API")
                    retries += 1
                    if retries < max_retries:
                        time.sleep(1)  # Wait before retrying
                        continue
                    else:
                        raise ValueError("Received empty response from OpenAI API after multiple attempts")
                
                # Log the first part of the response for debugging
                logger.info(f"AI response received, first 100 chars: {content[:100]}")
                return content
                
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                retries += 1
                if retries < max_retries:
                    logger.info(f"Retrying... ({retries}/{max_retries})")
                    time.sleep(1)  # Wait before retrying
                else:
                    raise
    
    async def generate_meal_plan(self, profile: UserProfile) -> DailyMealPlan:
        try:
            user_prompt = f"""Create a personalized daily meal plan based on these user details:
            Goal: {profile.primary_goal}
            Weight: {profile.weight_kg}kg
            Height: {profile.height_cm}cm
            Meat Eater: {profile.is_meat_eater}
            Lactose Intolerant: {profile.is_lactose_intolerant}
            Allergies: {', '.join(profile.allergies)}
            Eating Style: {profile.eating_style}
            Caffeine: {profile.caffeine_consumption}
            Sugar: {profile.sugar_consumption}

            You MUST respond with a valid JSON object containing personalized meal recommendations appropriate for this specific user. Return ONLY a JSON object matching this structure:
        
            {{
              "breakfast": {{
                "name": "[GENERATE APPROPRIATE NAME]",
                "description": "[GENERATE BRIEF DESCRIPTION]",
                "calories": [APPROPRIATE CALORIE NUMBER],
                "protein": [APPROPRIATE PROTEIN GRAMS],
                "carbs": [APPROPRIATE CARB GRAMS],
                "fat": [APPROPRIATE FAT GRAMS],
                "rationale": "[EXPLAIN WHY THIS MEAL FITS USER'S NEEDS]",
                "preparation_steps": ["[STEP 1]", "[STEP 2]", "..."]
              }},
              "lunch": {{
                "name": "[GENERATE APPROPRIATE NAME]",
                "description": "[GENERATE BRIEF DESCRIPTION]",
                "calories": [APPROPRIATE CALORIE NUMBER],
                "protein": [APPROPRIATE PROTEIN GRAMS],
                "carbs": [APPROPRIATE CARB GRAMS],
                "fat": [APPROPRIATE FAT GRAMS],
                "rationale": "[EXPLAIN WHY THIS MEAL FITS USER'S NEEDS]",
                "preparation_steps": ["[STEP 1]", "[STEP 2]", "..."]
              }},
              "snack": {{
                "name": "[GENERATE APPROPRIATE NAME]",
                "description": "[GENERATE BRIEF DESCRIPTION]",
                "calories": [APPROPRIATE CALORIE NUMBER],
                "protein": [APPROPRIATE PROTEIN GRAMS],
                "carbs": [APPROPRIATE CARB GRAMS],
                "fat": [APPROPRIATE FAT GRAMS],
                "rationale": "[EXPLAIN WHY THIS MEAL FITS USER'S NEEDS]",
                "preparation_steps": ["[STEP 1]", "[STEP 2]", "..."]
              }},
              "dinner": {{
                "name": "[GENERATE APPROPRIATE NAME]",
                "description": "[GENERATE BRIEF DESCRIPTION]",
                "calories": [APPROPRIATE CALORIE NUMBER],
                "protein": [APPROPRIATE PROTEIN GRAMS],
                "carbs": [APPROPRIATE CARB GRAMS],
                "fat": [APPROPRIATE FAT GRAMS],
                "rationale": "[EXPLAIN WHY THIS MEAL FITS USER'S NEEDS]",
                "preparation_steps": ["[STEP 1]", "[STEP 2]", "..."]
              }}
            }}
        
            IMPORTANT: 
            - Create realistic, nutritionally appropriate meals for this user's specific profile and goal
            - All nutritional values must be numbers without units (no "g" suffix)
            - Ensure preparation_steps is an array of strings with clear cooking/preparation instructions
            - Provide accurate nutritional values based on the ingredients
            - The response must be a valid JSON object with NO text outside the JSON
            - For a user trying to {profile.primary_goal}, adjust calories and macros accordingly
            - DO NOT include any markdown formatting, just return the raw JSON
            """

            # Get AI-generated meal plan content using the helper method
            content = await self._get_ai_response(user_prompt)
            
            # Log response for debugging
            logger.info(f"LLM response starts with: {content[:100]}...")
            
            # Try to clean the response if it's not valid JSON
            try:
                # First attempt to parse as is
                meal_plan_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON if it's embedded in text
                logger.warning("Initial JSON parsing failed, attempting to extract JSON from text")
                json_match = re.search(r'\{\s*"breakfast".*\}\s*\}', content, re.DOTALL)
                if json_match:
                    try:
                        content = json_match.group(0)
                        logger.info(f"Extracted JSON: {content[:100]}...")
                        meal_plan_data = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse extracted JSON: {e}")
                        logger.error(f"Raw content: {content}")
                        raise ValueError(f"Invalid JSON format from LLM: {e}")
                else:
                    logger.error("Could not extract JSON from response")
                    logger.error(f"Raw content: {content}")
                    raise ValueError("Invalid JSON format from LLM: Could not extract JSON pattern")
            
            # Validate required keys
            required_keys = ["breakfast", "lunch", "snack", "dinner"]
            for key in required_keys:
                if key not in meal_plan_data:
                    raise ValueError(f"Missing required key: {key}")
            
                # Validate meal object structure
                meal_keys = ["name", "description", "calories", "protein", "carbs", "fat", "rationale", "preparation_steps"]
                for meal_key in meal_keys:
                    if meal_key not in meal_plan_data[key]:
                        raise ValueError(f"Missing required key in {key}: {meal_key}")
        
            return DailyMealPlan(
                breakfast=self._create_meal_from_json(meal_plan_data["breakfast"]),
                lunch=self._create_meal_from_json(meal_plan_data["lunch"]),
                snack=self._create_meal_from_json(meal_plan_data["snack"]),
                dinner=self._create_meal_from_json(meal_plan_data["dinner"])
            )

        except Exception as e:
            logger.error(f"Error generating meal plan: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate meal plan: {str(e)}")