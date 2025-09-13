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
    title="Car Condition Analyzer API",
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
    """Корневой эндпоинт"""
    return {
        "message": "Car Condition Analyzer API",
        "version": "1.0.0",
        "description": "API для анализа состояния автомобилей",
        "endpoints": {
            "analyze": "/analyze - анализ изображения автомобиля",
            "health": "/health - проверка состояния API",
            "docs": "/docs - документация API"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния API"""
    try:
        status = {
            "status": "healthy",
            "image_processor": image_processor is not None,
            "condition_analyzer": condition_analyzer is not None,
            "models_loaded": {
                "damage_classifier": condition_analyzer.damage_classifier.model is not None if condition_analyzer else False,
                "dirt_classifier": condition_analyzer.dirt_classifier.model is not None if condition_analyzer else False,
                "scratch_classifier": condition_analyzer.scratch_classifier.model is not None if condition_analyzer else False
            }
        }
        return status
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze")
async def analyze_car_condition(file: UploadFile = File(...)):
    """
    Анализ состояния автомобиля по изображению
    
    Возвращает JSON с результатами анализа:
    - битый: 1 (битый) или 0 (не битый)
    - грязный: 1 (грязный) или 0 (чистый)
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
        
        # Детекция автомобиля на изображении
        car_detection = image_processor.detect_car_in_image(image)
        
        if not car_detection["has_car"]:
            return JSONResponse(
                status_code=200,
                content={
                    "битый": 0,
                    "грязный": 0,
                    "царапины": 0,
                    "message": "На изображении не обнаружен автомобиль",
                    "car_detected": False,
                    "analysis_performed": False
                }
            )
        
        # Если найден автомобиль, обрезаем изображение по области автомобиля
        if car_detection["detections"]:
            # Используем первое обнаружение с наибольшей уверенностью
            best_detection = max(car_detection["detections"], key=lambda x: x["confidence"])
            cropped_image = image_processor.crop_car_region(image, best_detection["bbox"])
        else:
            cropped_image = image
        
        # Предобработка изображения для классификации
        image_tensor = image_processor.preprocess_image(cropped_image)
        
        # Анализ состояния автомобиля
        analysis_result = condition_analyzer.analyze_car_condition(image_tensor)
        
        # Добавляем информацию о детекции автомобиля
        analysis_result["car_detected"] = True
        analysis_result["car_detection_confidence"] = best_detection["confidence"] if car_detection["detections"] else 1.0
        analysis_result["analysis_performed"] = True
        
        return JSONResponse(status_code=200, content=analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка анализа изображения: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/analyze-detailed")
async def analyze_car_condition_detailed(file: UploadFile = File(...)):
    """
    Подробный анализ состояния автомобиля по изображению
    
    Возвращает расширенную информацию включая уверенность модели и вероятности
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
        
        # Детекция автомобиля на изображении
        car_detection = image_processor.detect_car_in_image(image)
        
        if not car_detection["has_car"]:
            return JSONResponse(
                status_code=200,
                content={
                    "битый": 0,
                    "грязный": 0,
                    "царапины": 0,
                    "message": "На изображении не обнаружен автомобиль",
                    "car_detection": car_detection,
                    "detailed_analysis": None,
                    "analysis_performed": False
                }
            )
        
        # Обработка изображения для анализа
        if car_detection["detections"]:
            best_detection = max(car_detection["detections"], key=lambda x: x["confidence"])
            cropped_image = image_processor.crop_car_region(image, best_detection["bbox"])
        else:
            cropped_image = image
        
        # Предобработка изображения
        image_tensor = image_processor.preprocess_image(cropped_image)
        
        # Полный анализ состояния
        analysis_result = condition_analyzer.analyze_car_condition(image_tensor)
        
        # Добавляем подробную информацию о детекции
        analysis_result["car_detection"] = car_detection
        analysis_result["image_info"] = {
            "original_size": image.size,
            "processed_size": cropped_image.size if car_detection["detections"] else image.size,
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        return JSONResponse(status_code=200, content=analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка подробного анализа: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

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
        port=8000,
        reload=True,
        log_level="info"
    )