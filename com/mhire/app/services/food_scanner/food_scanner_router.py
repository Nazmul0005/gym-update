import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from com.mhire.app.services.food_scanner.food_scanner import FoodScanner
from com.mhire.app.services.food_scanner.food_scanner_schema import FoodScanResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Food Scanner"]
)

# Initialize Food Scanner
food_scanner = FoodScanner()

@router.post("/food-scanner", response_model=FoodScanResponse)
async def analyze_food(image: UploadFile = File(...)):
    """
    Analyze a food image and return detailed nutritional information including:
    - Food items identified
    - Nutritional content (calories, protein, carbs, fat)
    - Health benefits
    - Potential concerns
    - Serving suggestions
    """
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        analysis = await food_scanner.analyze_food_image(image)
        return FoodScanResponse(
            success=True,
            analysis=analysis
        )
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return FoodScanResponse(
            success=False,
            error=str(e)
        )