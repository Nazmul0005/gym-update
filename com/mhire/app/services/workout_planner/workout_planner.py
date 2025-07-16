import logging
import httpx
import json
from typing import List
import asyncio
from openai import OpenAI
from tavily import TavilyClient
from com.mhire.app.config.config import Config
from com.mhire.app.services.workout_planner.workout_planner_schema import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkoutPlanner:
    def __init__(self):
        config = Config()
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.model = config.model_name2
        self.tavily_api_key = config.tavily_api_key
        
        # Log the API key (first few characters for security)
        if self.tavily_api_key:
            logger.info(f"Tavily API key initialized: {self.tavily_api_key[:5]}... (length: {len(self.tavily_api_key)})")
        else:
            logger.warning("Tavily API key is not set or empty")
            
        # Initialize Tavily client only if API key is valid
        self.tavily_client = None  # Initialize to None by default
        if self.tavily_api_key and len(self.tavily_api_key) > 10:
            try:
                # Don't initialize the client here - we'll create a new one for each request
                # to avoid any potential authentication issues
                logger.info("Will create Tavily client per request")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {str(e)}")
        else:
            logger.warning("Tavily client not initialized due to missing or invalid API key")
        
    async def generate_workout_plan(self, profile: UserProfileRequest) -> WorkoutResponse:
        try:
            # Create workout structure for single day
            workout_structure = self._create_workout_structure(profile)
            
            # Generate single day workout
            focus = workout_structure["splits"][0]  # Use first split as the focus
            daily_workout = await self._generate_daily_workout(profile, focus, 1)
            
            return WorkoutResponse(
                success=True,
                workout_plan=[daily_workout],  # Single day workout in array
                error=None
            )
        except Exception as e:
            logger.error(f"Error generating workout plan: {str(e)}")
            return WorkoutResponse(
                success=False,
                workout_plan=[],
                error=str(e)
            )

    def _create_workout_structure(self, profile: UserProfileRequest) -> dict:
        # Base structure based on primary goal
        base_structures = {
            PrimaryGoal.BUILD_MUSCLE: {
                "splits": ["Upper Body Push"],
                "intensity": "High",
                "rest": "60-90s"
            },
            PrimaryGoal.LOSE_WEIGHT: {
                "splits": ["HIIT Cardio"],
                "intensity": "Moderate-High",
                "rest": "30-45s"
            },
            PrimaryGoal.EAT_HEALTHIER: {
                "splits": ["Full Body"],
                "intensity": "Moderate",
                "rest": "45-60s"
            }
        }
        
        structure = base_structures.get(profile.primary_goal)
        
        # Adjust based on dietary profile
        if profile.eating_style == EatingStyle.VEGAN or profile.eating_style == EatingStyle.VEGETARIAN:
            structure["nutrition_note"] = "Include pre-workout protein sources"
            
        # Adjust based on energy levels
        if profile.caffeine_consumption == ConsumptionFrequencyCaffine.NONE:
            structure["warm_up_duration"] = "15-20 minutes"  # Longer warm-up
        
        return structure

    async def _search_tavily_video(self, query: str) -> Optional[str]:
        """Search for exercise videos using Tavily API"""
        try:
            logging.info(f"Searching for video: {query}")
            
            # Check if API key is valid
            if not self.tavily_api_key or len(self.tavily_api_key) < 10:
                logger.warning("Invalid or missing Tavily API key")
                return None
            
            # Try both approaches - first the client library, then direct API call
            # Try the client library first
            try:
                # Create a new client for each request
                from tavily import TavilyClient
                client = TavilyClient(api_key=self.tavily_api_key)
                
                # Log the API key being used (first few chars only)
                logging.info(f"Using Tavily client with API key: {self.tavily_api_key[:5]}...")
                
                search_result = client.search(
                    query=f"{query} exercise video tutorial demonstration",
                    search_depth="advanced",
                    include_domains=["youtube.com"],
                    max_results=5
                )
                
                if search_result and search_result.get("results"):
                    videos = [r for r in search_result["results"] if "youtube.com" in r.get("url", "")]
                    if videos:
                        video_url = videos[0]["url"]
                        logging.info(f"Found video via client: {video_url}")
                        return video_url
            except Exception as e:
                logging.error(f"Error with Tavily client: {str(e)}. Trying direct API call...")
            
            # If client approach failed, try direct API call
            return await self._direct_tavily_api_call(query)
                
        except Exception as e:
            logging.error(f"Tavily API error for {query}: {str(e)}")
            # Return None instead of failing the entire workout generation
            return None
    
    async def _direct_tavily_api_call(self, query: str) -> Optional[str]:
        """Make a direct HTTP request to Tavily API"""
        try:
            url = "https://api.tavily.com/search"
            
            # Check if API key is valid
            if not self.tavily_api_key or len(self.tavily_api_key) < 10:
                logger.warning("Invalid or missing Tavily API key for direct API call")
                return None
            
            # Try multiple authentication approaches
            # First attempt: Use the API key directly with Bearer prefix
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.tavily_api_key}"
            }
            
            payload = {
                "query": f"{query} exercise video tutorial demonstration",
                "search_depth": "advanced",
                "include_domains": ["youtube.com"],
                "max_results": 5
            }
            
            logging.info("Making direct API call to Tavily with Bearer token")
            
            # Try multiple authentication approaches
            async with httpx.AsyncClient() as client:
                # First try with Bearer token
                response = await client.post(url, headers=headers, json=payload)
                logging.info(f"Tavily API response status with Bearer token: {response.status_code}")
                
                # If that fails, try with API key as a parameter
                if response.status_code == 401:
                    logging.info("Bearer token authentication failed, trying with api_key parameter")
                    # Second attempt: Use the API key as a parameter
                    payload["api_key"] = self.tavily_api_key
                    headers = {"Content-Type": "application/json"}
                    response = await client.post(url, headers=headers, json=payload)
                    logging.info(f"Tavily API response status with api_key parameter: {response.status_code}")
                
                # If that fails, try with a different Bearer format (without tvly- prefix)
                if response.status_code == 401 and self.tavily_api_key.startswith("tvly-"):
                    logging.info("API key parameter failed, trying with modified Bearer token (without tvly- prefix)")
                    # Third attempt: Use the API key without the tvly- prefix
                    api_key_without_prefix = self.tavily_api_key[5:] if self.tavily_api_key.startswith("tvly-") else self.tavily_api_key
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key_without_prefix}"
                    }
                    response = await client.post(url, headers=headers, json=payload)
                    logging.info(f"Tavily API response status with modified Bearer token: {response.status_code}")
                
                # If all authentication attempts fail, try one last approach with X-Api-Key header
                if response.status_code == 401:
                    logging.info("All previous authentication methods failed, trying with X-Api-Key header")
                    # Fourth attempt: Use X-Api-Key header
                    headers = {
                        "Content-Type": "application/json",
                        "X-Api-Key": self.tavily_api_key
                    }
                    response = await client.post(url, headers=headers, json=payload)
                    logging.info(f"Tavily API response status with X-Api-Key header: {response.status_code}")
                
                if response.status_code == 200:
                    search_result = response.json()
                    if search_result and search_result.get("results"):
                        videos = [r for r in search_result["results"] if "youtube.com" in r.get("url", "")]
                        if videos:
                            video_url = videos[0]["url"]
                            logging.info(f"Found video via direct API: {video_url}")
                            return video_url
                else:
                    logging.error(f"All authentication methods failed. Final status code: {response.status_code}")
                    logging.error(f"Response text: {response.text}")
            
            # If we reach here, no videos were found or all authentication methods failed
            logging.warning(f"No suitable video found for: {query} using direct API call")
            return None
        except Exception as e:
            logging.error(f"Error in direct Tavily API call: {str(e)}")
            return None
            
    async def _get_ai_response(self, prompt: str) -> str:
        """Get workout plan from OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional fitness coach creating detailed workout plans. Always provide exactly 3 exercises for each section (warm-up, main routine, cool-down). No more, no less."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    async def _generate_daily_workout(self, profile: UserProfileRequest, focus: str, day: int) -> DailyWorkout:
        try:
            # Get AI-generated workout content
            prompt = self._create_workout_prompt(profile, focus, day)
            workout_content = await self._get_ai_response(prompt)
            
            # Parse the workout data first
            workout_data = self._parse_workout_response(workout_content)
            
            # Add individual exercise videos and durations
            await self._add_exercise_videos_and_durations(workout_data)
            
            # Calculate total duration for each segment based on exercise durations
            warm_up_duration = self._calculate_segment_duration(workout_data["warm_up"])
            main_routine_duration = self._calculate_segment_duration(workout_data["main_routine"])
            cool_down_duration = self._calculate_segment_duration(workout_data["cool_down"])
            
            return DailyWorkout(
                day=f"Day {day}",
                focus=focus,
                warm_up=WorkoutSegment(
                    motto="Keep moving—you've got this.",
                    exercises=workout_data["warm_up"],
                    duration=warm_up_duration,
                    video_url=None
                ),
                main_routine=WorkoutSegment(
                    motto="You're doing awesome—keep the energy up.",
                    exercises=workout_data["main_routine"],
                    duration=main_routine_duration,
                    video_url=None
                ),
                cool_down=WorkoutSegment(
                    motto="Breathe in peace—breathe out strength.",
                    exercises=workout_data["cool_down"],
                    duration=cool_down_duration,
                    video_url=None
                )
            )
        except Exception as e:
            logger.error(f"Error generating daily workout: {str(e)}")
            raise

    async def _add_exercise_videos_and_durations(self, workout_data: dict):
        """Search for individual exercise videos and add AI-generated durations"""
        try:
            for section_name, exercises in workout_data.items():
                for exercise in exercises:
                    # Search for individual exercise video
                    exercise_video = await self._search_tavily_video(f"{exercise.name} exercise tutorial")
                    exercise.video_url = exercise_video if exercise_video else ""
                    
                    # Generate duration using AI
                    duration = await self._generate_exercise_duration(exercise, section_name)
                    exercise.duration = duration
                    
        except Exception as e:
            logger.error(f"Error adding exercise videos and durations: {str(e)}")
            # Don't fail the entire workout generation, just log the error
            
    async def _generate_exercise_duration(self, exercise: Exercise, section_type: str) -> int:
        """Generate appropriate duration for individual exercise using AI and convert to seconds"""
        try:
            prompt = f"""
            Generate an appropriate duration for this {section_type} exercise:
            Exercise: {exercise.name}
            Sets: {exercise.sets}
            Reps: {exercise.reps}
            Rest: {exercise.rest}
            
            Provide ONLY the duration in format like "2-3 minutes", "45 seconds", "1 minute", etc.
            Consider the exercise type and intensity level.
            """
            
            duration_response = await self._get_ai_response(prompt)
            # Extract just the duration from the response (in case AI adds extra text)
            duration_str = duration_response.strip().split('\n')[0].strip()
            
            # Convert the duration string to seconds
            return self._convert_duration_to_seconds(duration_str)
            
        except Exception as e:
            logger.error(f"Error generating exercise duration: {str(e)}")
            # Return default duration based on section type in seconds
            if section_type == "warm_up":
                return 120  # 2 minutes
            elif section_type == "main_routine":
                return 210  # 3.5 minutes (average of 3-4 minutes)
            else:  # cool_down
                return 90   # 1.5 minutes (average of 1-2 minutes)

    def _calculate_segment_duration(self, exercises: List[Exercise]) -> int:
        """Calculate the total duration of a workout segment by summing exercise durations"""
        total_duration = 0
        for exercise in exercises:
            # If exercise has a duration, add it to the total
            if exercise.duration is not None:
                total_duration += exercise.duration
            else:
                # Default duration if not set
                total_duration += 120  # 2 minutes default
        
        # Ensure minimum duration of 60 seconds
        return max(60, total_duration)
    
    def _convert_duration_to_seconds(self, duration_str: str) -> int:
        """Convert a duration string like '2-3 minutes' or '45 seconds' to seconds"""
        if not duration_str:
            return 120  # Default to 2 minutes if empty
            
        duration_str = duration_str.lower()
        
        # Handle ranges like '2-3 minutes' by taking the average
        if '-' in duration_str:
            parts = duration_str.split('-')
            if len(parts) == 2:
                # Extract the numeric values
                try:
                    # Try to extract first number
                    first_num = ''.join(c for c in parts[0] if c.isdigit() or c == '.')
                    # Try to extract second number
                    second_num = ''.join(c for c in parts[1] if c.isdigit() or c == '.')
                    
                    if first_num and second_num:
                        first_val = float(first_num)
                        second_val = float(second_num)
                        avg_val = (first_val + second_val) / 2
                        
                        # Determine the unit (minutes or seconds)
                        if 'minute' in duration_str or 'min' in duration_str:
                            return int(avg_val * 60)  # Convert minutes to seconds
                        else:
                            return int(avg_val)  # Assume seconds
                except (ValueError, TypeError):
                    pass  # Fall through to the next parsing method
        
        # Handle simple durations like '45 seconds' or '2 minutes'
        try:
            # Extract numeric part
            numeric_part = ''.join(c for c in duration_str if c.isdigit() or c == '.')
            if numeric_part:
                value = float(numeric_part)
                
                # Determine the unit
                if 'minute' in duration_str or 'min' in duration_str:
                    return int(value * 60)  # Convert minutes to seconds
                else:
                    return int(value)  # Assume seconds
        except (ValueError, TypeError):
            pass
            
        # Default fallback
        return 120  # Default to 2 minutes if parsing fails
    
    def _create_workout_prompt(self, profile: UserProfileRequest, focus: str, day: int) -> str:
        return f"""Create a detailed {focus} workout for Day {day} considering:
        User Profile:
        - Primary Goal: {profile.primary_goal}
        - Weight: {profile.weight_kg}kg
        - Height: {profile.height_cm}cm
        - Diet: {profile.eating_style}
        - Meat Eater: {profile.is_meat_eater}
        - Lactose Intolerant: {profile.is_lactose_intolerant}
        - Allergies: {', '.join(profile.allergies)}
        - Caffeine: {profile.caffeine_consumption}
        - Sugar: {profile.sugar_consumption}

        CRITICAL REQUIREMENTS: 
        1. You MUST provide EXACTLY 3 exercises for each section. No more, no less.
        2. Instructions must be ONE LINE and CONCISE (maximum 10-15 words).
        If you provide fewer than 3 exercises or more than 3 exercises in any section, the system will fail.

        Provide the workout plan in this EXACT format:
        
        Warm-up:
        - [Exercise Name] | [One line concise instruction]
        - [Exercise Name] | [One line concise instruction] 
        - [Exercise Name] | [One line concise instruction]
        
        Main Routine:
        - [Exercise Name] | Sets: [X] | Reps: [X] | Rest: [Xs] | [One line concise instruction]
        - [Exercise Name] | Sets: [X] | Reps: [X] | Rest: [Xs] | [One line concise instruction]
        - [Exercise Name] | Sets: [X] | Reps: [X] | Rest: [Xs] | [One line concise instruction]
        
        Cool-down:
        - [Exercise Name] | [One line concise instruction]
        - [Exercise Name] | [One line concise instruction]
        - [Exercise Name] | [One line concise instruction]
        
        Note: Keep all instructions brief and actionable. For warm-up and cool-down, provide only exercise name and concise instructions.
        """

    def _parse_workout_response(self, content: str) -> dict:
        """Parse the AI-generated workout content into structured segments with exactly 3 exercises each"""
        segments = {
            "warm_up": [],
            "main_routine": [],
            "cool_down": []
        }
        current_section = None
        
        try:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            for line in lines:
                lower_line = line.lower()
                
                # Detect section headers
                if "warm-up:" in lower_line or "warmup:" in lower_line:
                    current_section = "warm_up"
                    continue
                elif "main routine:" in lower_line or "main workout:" in lower_line:
                    current_section = "main_routine"
                    continue
                elif "cool-down:" in lower_line or "cooldown:" in lower_line:
                    current_section = "cool_down"
                    continue
                
                # Skip lines that don't start with bullet point or dash
                if not line.lstrip().startswith(('-', '•', '*')):
                    continue
                
                # Only process if we're in a valid section and haven't reached 3 exercises
                if current_section and len(segments[current_section]) < 3:
                    try:
                        # Split on pipe and clean up each part
                        parts = [p.strip() for p in line.lstrip('- •*').split('|')]
                        
                        # Extract exercise name (required)
                        name = parts[0].strip() if parts else f"Exercise {len(segments[current_section]) + 1}"
                        
                        if current_section == "main_routine":
                            # Parse main routine with more detailed info
                            exercise_data = {
                                'sets': 3,  # Default values
                                'reps': '10-12',
                                'rest': '60s',
                                'instructions': 'Perform with proper form'
                            }
                            
                            # Process each part looking for specific keywords
                            for part in parts[1:]:
                                part_lower = part.lower().strip()
                                if 'sets:' in part_lower:
                                    sets_str = ''.join(filter(str.isdigit, part))
                                    exercise_data['sets'] = int(sets_str) if sets_str else 3
                                elif 'reps:' in part_lower:
                                    exercise_data['reps'] = part.split(':')[-1].strip()
                                elif 'rest:' in part_lower:
                                    exercise_data['rest'] = part.split(':')[-1].strip()
                                else:
                                    if len(part.strip()) > 10:  # Only use meaningful instructions
                                        instructions = part.strip()
                                        # Ensure instructions are concise (limit to ~60 characters)
                                        if len(instructions) > 60:
                                            instructions = instructions[:57] + "..."
                                        exercise_data['instructions'] = instructions
                            
                            # Create exercise with extracted or default values
                            exercise = Exercise(
                                name=name,
                                sets=exercise_data['sets'],
                                reps=exercise_data['reps'],
                                rest=exercise_data['rest'],
                                instructions=exercise_data['instructions'],
                                video_url="",  # Will be populated later
                                duration=None  # Will be populated later with seconds
                            )
                        else:
                            # Simpler parsing for warm-up and cool-down
                            instructions = parts[1].strip() if len(parts) > 1 else "Perform with proper form"
                            
                            # Ensure instructions are concise (limit to ~60 characters)
                            if len(instructions) > 60:
                                instructions = instructions[:57] + "..."
                            
                            # Set appropriate default values for warm-up and cool-down
                            if current_section == "warm_up":
                                reps_value = "10-15 reps"
                                rest_value = "30 seconds"
                            else:  # cool_down
                                reps_value = "Hold for 30 seconds"
                                rest_value = "15 seconds"
                            
                            exercise = Exercise(
                                name=name,
                                sets=1,
                                reps=reps_value,
                                rest=rest_value,
                                instructions=instructions,
                                video_url="",  # Will be populated later
                                duration=None  # Will be populated later with seconds
                            )
                        
                        segments[current_section].append(exercise)
                    except Exception as e:
                        logger.warning(f"Error parsing exercise line '{line}': {str(e)}")
                        continue
            
            # Validate that each section has exactly 3 exercises
            for section in segments:
                if len(segments[section]) != 3:
                    logger.error(f"Section '{section}' has {len(segments[section])} exercises instead of 3")
                    raise ValueError(f"Failed to parse exactly 3 exercises for {section} section")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error parsing workout response: {str(e)}")
            raise ValueError(f"Failed to parse AI workout response: {str(e)}")