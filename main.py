from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from PIL import Image
import io

from utils import ImageProcessor
from model import CarConditionAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="inDrive API",
    description="API для анализа состояния автомобилей: определение повреждений, загрязнений и царапин",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для моделей
image_processor = None
condition_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Инициализация моделей при запуске приложения"""
    global image_processor, condition_analyzer
    
    try:
        logger.info("Инициализация системы анализа автомобилей...")
        
        # Инициализация процессора изображений
        image_processor = ImageProcessor()
        
        # Инициализация анализатора состояния автомобилей
        condition_analyzer = CarConditionAnalyzer()
        
        logger.info("Система успешно инициализирована!")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации системы: {e}")
        raise e

@app.get("/")
async def root():
    """Анализ состояния автомобилей"""
    return {
        "message": "Car Condition Analyzer API",
        "endpoints": {
            "analyze": "/analyze - анализ состояния автомобиля",
            "analyze_by_parts": "/analyze-by-parts - анализ по частям", 
            "health": "/health - состояние системы"
        },
        "result_format": {
            "битый": "1 (есть повреждения) или 0 (нет повреждений)",
            "грязный": "1 (загрязненный) или 0 (чистый)",
            "царапины": "1 (есть царапины) или 0 (нет царапин)"
        }
    }


@app.post("/analyze")
async def analyze_car_condition(file: UploadFile = File(...)):
    """
    Анализ состояния автомобиля
    
    Возвращает:
    - битый: 1 (есть повреждения) или 0 (нет повреждений)
    - грязный: 1 (загрязненный) или 0 (чистый) 
    - царапины: 1 (есть царапины) или 0 (нет царапин)
    """
    if not image_processor or not condition_analyzer:
        raise HTTPException(status_code=503, detail="Система анализа не инициализирована")
    
    try:
        # Чтение файла
        contents = await file.read()
        
        # Валидация изображения
        is_valid, message, image = image_processor.validate_image(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Предобработка изображения для классификации
        image_tensor = image_processor.preprocess_image(image)
        
        # Анализ состояния автомобиля
        analysis_result = condition_analyzer.analyze_car_condition(image_tensor)
        
        # Возвращаем только основную информацию
        simple_result = {
            "битый": analysis_result["битый"],
            "грязный": analysis_result["грязный"],
            "царапины": analysis_result["царапины"]
        }
        
        return JSONResponse(status_code=200, content=simple_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка анализа изображения: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/analyze-by-parts")
async def analyze_car_parts(file: UploadFile = File(...)):
    """
    Анализ по частям автомобиля
    
    Возвращает результаты для каждой части отдельно:
    - Общий результат (битый: 0/1, грязный: 0/1, царапины: 0/1)
    - Детализация по частям с указанием конкретных мест
    """
    if not image_processor or not condition_analyzer:
        raise HTTPException(status_code=503, detail="Система анализа не инициализирована")
    
    try:
        # Чтение файла
        contents = await file.read()
        
        # Валидация изображения
        is_valid, message, image = image_processor.validate_image(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Сегментация изображения на части
        segments = image_processor.segment_car_parts(image)
        
        # Анализ по частям
        analysis_result = condition_analyzer.analyze_by_parts(segments)
        
        # Упрощенный результат
        parts_details = {}
        if "parts_analysis" in analysis_result:
            for part_name, part_data in analysis_result["parts_analysis"].items():
                parts_details[part_name] = {
                    "битый": part_data.get("damaged", 0),
                    "грязный": part_data.get("dirty", 0),
                    "царапины": part_data.get("scratched", 0)
                }
        
        simple_result = {
            "битый": analysis_result["битый"],
            "грязный": analysis_result["грязный"], 
            "царапины": analysis_result["царапины"],
            "части": parts_details
        }
        
        return JSONResponse(status_code=200, content=simple_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка анализа по частям: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# Удален детальный endpoint - оставляем только основную функциональность

@app.get("/models/status")
async def get_models_status():
    """Получение статуса загруженных моделей"""
    if not condition_analyzer:
        return {"status": "not_initialized"}
    
    return {
        "damage_classifier": {
            "loaded": condition_analyzer.damage_classifier.model is not None,
            "model_path": condition_analyzer.damage_classifier.model_path,
            "classes": condition_analyzer.damage_classifier.class_names
        },
        "dirt_classifier": {
            "loaded": condition_analyzer.dirt_classifier.model is not None,
            "model_path": condition_analyzer.dirt_classifier.model_path,
            "classes": condition_analyzer.dirt_classifier.class_names
        },
        "scratch_classifier": {
            "loaded": condition_analyzer.scratch_classifier.model is not None,
            "model_path": condition_analyzer.scratch_classifier.model_path,
            "classes": condition_analyzer.scratch_classifier.class_names
        }
    }

if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info"
    )