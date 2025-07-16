import logging
import re
import base64
from fastapi import HTTPException, UploadFile
from openai import OpenAI
from com.mhire.app.config.config import Config
from com.mhire.app.services.food_scanner.food_scanner_schema import FoodScanResponse, FoodAnalysis, NutritionInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodScanner:
    def __init__(self):
        try:
            config = Config()
            self.client = OpenAI(api_key=config.openai_api_key)
            self.model = config.model_name
        except Exception as e:
            logger.error(f"Error initializing FoodScanner: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Food Scanner: {str(e)}")

    async def analyze_food_image(self, image: UploadFile) -> FoodAnalysis:
        try:
            # Read image content
            image_content = await image.read()
            
            # Get image type - if content_type not available, default to jpeg
            content_type = image.content_type if image.content_type else "image/jpeg"
            
            # Ensure the content type is correctly formatted
            if "/" not in content_type:
                content_type = f"image/{content_type}"
            
            # Generate base64 encoding of the image
            base64_image = base64.b64encode(image_content).decode('utf-8')
            
            # Log diagnostic info
            logger.info(f"Processing image with content type: {content_type}")
            logger.info(f"Image size: {len(image_content)} bytes")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a professional nutritionist and food analyst specializing in visual food analysis. For ANY food image (simple or complex):

                            1. First, identify ALL ingredients and components
                            2. Then, considering the COMPLETE dish, provide TOTAL nutritional values
                            3. Follow this EXACT format:

                            FOOD ITEMS AND INGREDIENTS:
                            - [Dish name]
                            - [List all visible ingredients]
                            - [List garnishes/sides if any]

                            TOTAL NUTRITIONAL VALUES:
                            Calories: [X] kcal
                            Protein: [X] g
                            Carbohydrates: [X] g
                            Fat: [X] g

                            HEALTH BENEFITS:
                            - [List key benefits]

                            DIETARY CONCERNS:
                            - [List allergens or concerns]

                            IMPORTANT RULES:
                            - ALWAYS analyze the COMPLETE dish
                            - ALWAYS provide numerical values
                            - If exact values unknown, provide educated estimates
                            - Keep responses focused and concise
                            - For complex dishes, provide ONE total nutritional value"""
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this food image and provide total nutritional values. If exact values are unknown, provide your best estimates based on visual analysis."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                )
            except Exception as api_error:
                logger.error(f"OpenAI API error: {str(api_error)}")
                raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(api_error)}")
            
            # Extract the analysis text
            analysis_text = response.choices[0].message.content.strip()
            
            if not analysis_text:
                raise HTTPException(status_code=500, detail="Food analysis returned empty results")
            
            # Log the first part of the raw analysis text for debugging
            logger.info(f"Raw analysis text (first 200 chars): {analysis_text[:200]}...")
            
            # Parse the response to extract structured information
            parsed_info = self._parse_analysis(analysis_text)
            
            # Additional validation to ensure we have values
            if parsed_info["calories"] == 0 or parsed_info["protein"] == 0:
                raise HTTPException(status_code=500, detail="Failed to extract nutritional values from analysis")
            
            return FoodAnalysis(
                food_items=parsed_info["food_items"],
                nutrition=NutritionInfo(
                    calories=parsed_info["calories"],
                    protein=parsed_info["protein"],
                    carbs=parsed_info["carbs"],
                    fat=parsed_info["fat"]
                ),
                health_benefits=parsed_info["health_benefits"],
                concerns=parsed_info["concerns"]
            )
        except Exception as e:
            logger.error(f"Error analyzing food image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to analyze food image: {str(e)}")

    def _parse_analysis(self, text: str) -> dict:
        """Parse the AI response with improved nutrition value extraction"""    
        result = {
            "food_items": [],
            "calories": 0.0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "health_benefits": [],
            "concerns": []
        }
        
        # Split text into sections using headers
        sections = {}
        current_section = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            lower_line = line.lower()
            if "food items and ingredients:" in lower_line:
                current_section = "items"
            elif "total nutritional values" in lower_line:
                current_section = "nutrition"
            elif "health benefits:" in lower_line:
                current_section = "benefits"
            elif "dietary concerns:" in lower_line:
                current_section = "concerns"
            
            # Add content to sections
            if current_section and current_section not in sections:
                sections[current_section] = []
            if current_section:
                sections[current_section].append(line)
        
        # Process food items
        if "items" in sections:
            for line in sections["items"]:
                if line.strip().startswith(("-", "•", "*", "○")):
                    item = line.lstrip("- •*○").strip()
                    if item and not any(x.lower() in item.lower() for x in ["food items", "ingredients"]):
                        result["food_items"].append(item)
        
        # Process nutrition values
        if "nutrition" in sections:
            for line in sections["nutrition"]:
                if ":" in line:
                    key, value = [x.strip() for x in line.split(":", 1)]
                    key = key.lower()
                    if "calories" in key:
                        result["calories"] = self._extract_number(value)
                    elif "protein" in key:
                        result["protein"] = self._extract_number(value)
                    elif "carbs" in key or "carbohydrates" in key:
                        result["carbs"] = self._extract_number(value)
                    elif "fat" in key:
                        result["fat"] = self._extract_number(value)
        
        # Process other sections
        if "benefits" in sections:
            for line in sections["benefits"]:
                if line.strip().startswith(("-", "•", "*", "○")):
                    benefit = line.lstrip("- •*○").strip()
                    if benefit:
                        result["health_benefits"].append(benefit)
        
        if "concerns" in sections:
            for line in sections["concerns"]:
                if line.strip().startswith(("-", "•", "*", "○")):
                    concern = line.lstrip("- •*○").strip()
                    if concern:
                        result["concerns"].append(concern)
        
        # Validate nutritional values
        if result["calories"] == 0 or result["protein"] == 0 or result["carbs"] == 0 or result["fat"] == 0:
            raise ValueError("Failed to extract complete nutritional information from the analysis")
        
        return result

    def _extract_number(self, text: str) -> float:
        """Extract the first number from text, handling various formats"""
        matches = re.findall(r'(\d+(?:\.\d+)?)', text)
        return float(matches[0]) if matches else 0.0