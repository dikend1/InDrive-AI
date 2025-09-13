import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from typing import Tuple, Optional, List, Dict
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Класс для обработки изображений и детекции автомобилей"""
    
    def __init__(self, car_detection_model_path: str = "yolov8n.pt"):
        """
        Инициализация процессора изображений
        
        Args:
            car_detection_model_path: путь к модели YOLO для детекции автомобилей
        """
        try:
            self.car_detector = YOLO(car_detection_model_path)
            logger.info(f"Модель детекции автомобилей загружена: {car_detection_model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели детекции: {e}")
            self.car_detector = None
    
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Предобработка изображения для модели классификации
        
        Args:
            image: PIL изображение
            target_size: целевой размер изображения
            
        Returns:
            Тензор для подачи в модель
        """
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        tensor = transform(image)
        return tensor.unsqueeze(0)  # Добавляем batch dimension
    
    def detect_car_in_image(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        """
        Детекция автомобилей на изображении
        
        Args:
            image: PIL изображение
            confidence_threshold: порог уверенности для детекции
            
        Returns:
            Словарь с результатами детекции
        """
        if self.car_detector is None:
            return {"has_car": False, "message": "Модель детекции не загружена"}
        
        try:
            # Конвертируем PIL в numpy array для YOLO
            image_np = np.array(image)
            if image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            results = self.car_detector(image_np, conf=confidence_threshold, verbose=False)
            
            car_detected = False
            car_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Класс 2 в COCO dataset - это автомобиль
                        if int(box.cls) == 2:  # car class
                            car_detected = True
                            confidence = float(box.conf)
                            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            car_boxes.append({
                                "bbox": bbox,
                                "confidence": confidence
                            })
            
            return {
                "has_car": car_detected,
                "car_count": len(car_boxes),
                "detections": car_boxes,
                "message": "Автомобиль обнаружен" if car_detected else "Автомобиль не обнаружен"
            }
            
        except Exception as e:
            logger.error(f"Ошибка детекции автомобиля: {e}")
            return {"has_car": False, "message": f"Ошибка детекции: {str(e)}"}
    
    def crop_car_region(self, image: Image.Image, bbox: List[float], padding: float = 0.1) -> Image.Image:
        """
        Обрезка изображения по области автомобиля
        
        Args:
            image: PIL изображение
            bbox: координаты бounding box [x1, y1, x2, y2]
            padding: отступ от границ bbox (доля от размера)
            
        Returns:
            Обрезанное изображение
        """
        width, height = image.size
        x1, y1, x2, y2 = bbox
        
        # Добавляем отступы
        w = x2 - x1
        h = y2 - y1
        pad_w = w * padding
        pad_h = h * padding
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)
        
        return image.crop((x1, y1, x2, y2))
    
    def validate_image(self, image_data: bytes) -> Tuple[bool, str, Optional[Image.Image]]:
        """
        Валидация загруженного изображения
        
        Args:
            image_data: байты изображения
            
        Returns:
            Кортеж (валидно, сообщение, PIL изображение или None)
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Проверяем формат
            if image.format not in ['JPEG', 'PNG', 'JPG']:
                return False, "Неподдерживаемый формат изображения. Используйте JPEG или PNG.", None
            
            # Проверяем размер файла (макс 10MB)
            if len(image_data) > 10 * 1024 * 1024:
                return False, "Файл слишком большой. Максимальный размер: 10MB.", None
            
            # Проверяем размеры изображения
            if image.size[0] < 100 or image.size[1] < 100:
                return False, "Изображение слишком маленькое. Минимальный размер: 100x100px.", None
                
            if image.size[0] > 4096 or image.size[1] > 4096:
                return False, "Изображение слишком большое. Максимальный размер: 4096x4096px.", None
            
            return True, "Изображение валидно", image
            
        except Exception as e:
            return False, f"Ошибка обработки изображения: {str(e)}", None

def enhance_image_for_classification(image: Image.Image) -> Image.Image:
    """
    Улучшение изображения для лучшей классификации
    
    Args:
        image: PIL изображение
        
    Returns:
        Улучшенное изображение
    """
    # Конвертируем в numpy для обработки с OpenCV
    image_np = np.array(image)
    
    # Улучшение контрастности
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced)

def resize_and_pad_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Изменение размера изображения с сохранением пропорций и добавлением отступов
    
    Args:
        image: PIL изображение
        target_size: целевой размер (ширина, высота)
        
    Returns:
        Измененное изображение
    """
    target_width, target_height = target_size
    
    # Вычисляем коэффициент масштабирования
    scale = min(target_width / image.width, target_height / image.height)
    
    # Новые размеры с сохранением пропорций
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    
    # Изменяем размер
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Создаем новое изображение с целевым размером
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # Серый фон
    
    # Вычисляем позицию для центрирования
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # Вставляем изображение в центр
    new_image.paste(resized, (paste_x, paste_y))
    
    return new_image