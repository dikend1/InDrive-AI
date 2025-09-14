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
        
        # Определяем типы областей автомобиля для анализа
        self.car_parts_mapping = {
            'exterior': ['кузов', 'бампер', 'крыло', 'дверь', 'капот', 'багажник'],
            'interior': ['салон', 'сиденье', 'панель', 'руль', 'приборы'],
            'wheels': ['колесо', 'диск', 'шина'],
            'lights': ['фара', 'фонарь', 'поворотник']
        }
    
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
    
    def analyze_image_content(self, image: Image.Image) -> Dict:
        """
        Анализ содержимого изображения для определения видимых частей автомобиля
        
        Args:
            image: PIL изображение
            
        Returns:
            Словарь с информацией о видимых частях
        """
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        
        # Анализируем позицию и содержимое изображения
        analysis = {
            'image_type': 'unknown',
            'visible_parts': [],
            'analysis_regions': {},
            'recommended_models': []
        }
        
        # Простые эвристики для определения типа изображения
        # В реальной системе здесь был бы более сложный анализ
        
        # Проверяем соотношение сторон
        aspect_ratio = width / height
        
        # Анализ цветовых особенностей для определения салона vs экстерьера
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Анализ текстур и паттернов
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Определяем тип изображения по признакам
        if self._is_interior_image(image_np, hsv, edge_density):
            analysis['image_type'] = 'interior'
            analysis['visible_parts'] = ['салон', 'сиденья', 'панель_приборов']
            analysis['recommended_models'] = ['interior_dirt', 'interior_damage', 'interior_scratch']
            analysis['analysis_regions'] = self._get_interior_regions(width, height)
        else:
            analysis['image_type'] = 'exterior'
            analysis['visible_parts'] = ['кузов', 'бампер', 'капот']
            analysis['recommended_models'] = ['exterior_dirt', 'exterior_damage', 'exterior_scratch']
            analysis['analysis_regions'] = self._get_exterior_regions(width, height)
        
        return analysis
    
    def _is_interior_image(self, image_np: np.ndarray, hsv: np.ndarray, edge_density: float) -> bool:
        """Определяет, является ли изображение салоном автомобиля"""
        # Анализ цветовых характеристик
        # Салон обычно имеет более темные тона и меньше ярких цветов
        v_channel = hsv[:, :, 2]
        brightness = np.mean(v_channel)
        
        # Салон обычно имеет больше прямых линий и геометрических форм
        lines = cv2.HoughLinesP(cv2.Canny(cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY), 50, 150),
                               1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        line_count = len(lines) if lines is not None else 0
        
        # Простая эвристика: салон = низкая яркость + много линий + высокая плотность краев
        return brightness < 120 and line_count > 10 and edge_density > 0.1
    
    def _get_interior_regions(self, width: int, height: int) -> Dict:
        """Определяет регионы для анализа салона"""
        return {
            'dashboard': {'x1': 0, 'y1': 0, 'x2': width, 'y2': height//3},
            'seats': {'x1': 0, 'y1': height//3, 'x2': width, 'y2': 2*height//3},
            'floor': {'x1': 0, 'y1': 2*height//3, 'x2': width, 'y2': height}
        }
    
    def _get_exterior_regions(self, width: int, height: int) -> Dict:
        """Определяет регионы для анализа экстерьера"""
        return {
            'upper': {'x1': 0, 'y1': 0, 'x2': width, 'y2': height//3},      # крыша, капот
            'middle': {'x1': 0, 'y1': height//3, 'x2': width, 'y2': 2*height//3},  # двери, кузов
            'lower': {'x1': 0, 'y1': 2*height//3, 'x2': width, 'y2': height}       # бампер, пороги
        }
    
    def segment_car_parts(self, image: Image.Image, car_bbox: List[float] = None) -> Dict:
        """
        Сегментация изображения автомобиля на отдельные части для целевого анализа
        
        Args:
            image: PIL изображение
            car_bbox: bounding box автомобиля (опционально)
            
        Returns:
            Словарь с сегментированными областями
        """
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        
        # Анализируем содержимое изображения
        content_analysis = self.analyze_image_content(image)
        
        segments = {
            'image_type': content_analysis['image_type'],
            'regions': {},
            'cropped_regions': {}
        }
        
        # Получаем регионы в зависимости от типа изображения
        regions = content_analysis['analysis_regions']
        
        for region_name, coords in regions.items():
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Обрезаем регион
            cropped_region = image.crop((x1, y1, x2, y2))
            
            segments['regions'][region_name] = {
                'bbox': [x1, y1, x2, y2],
                'size': (x2-x1, y2-y1),
                'area_percentage': ((x2-x1) * (y2-y1)) / (width * height) * 100
            }
            
            segments['cropped_regions'][region_name] = cropped_region
        
        return segments
    
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