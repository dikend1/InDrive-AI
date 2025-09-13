import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CarConditionClassifier:
    """Базовый класс для классификации состояния автомобиля"""
    
    def __init__(self, model_path: str, num_classes: int = 2, model_type: str = "resnet18"):
        """
        Инициализация классификатора
        
        Args:
            model_path: путь к файлу модели
            num_classes: количество классов
            model_type: тип модели (resnet18, resnet50, etc.)
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        
        self._load_model()
    
    def _create_model(self) -> nn.Module:
        """Создание архитектуры модели"""
        if self.model_type == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_type == "resnet50":
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_type == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")
        
        return model
    
    def _load_model(self):
        """Загрузка предобученной модели"""
        try:
            if os.path.exists(self.model_path):
                self.model = self._create_model()
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Обработка различных форматов сохранения
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    elif 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Модель загружена: {self.model_path}")
            else:
                logger.warning(f"Файл модели не найден: {self.model_path}")
                # Создаем простую модель-заглушку для демонстрации
                self.model = self._create_model()
                self.model.to(self.device)
                self.model.eval()
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {self.model_path}: {e}")
            # Создаем модель-заглушку
            self.model = self._create_model()
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        """
        Предсказание класса изображения
        
        Args:
            image_tensor: тензор изображения
            
        Returns:
            Словарь с результатами предсказания
        """
        if self.model is None:
            return {"confidence": 0.5, "predicted_class": 0, "probabilities": [0.5, 0.5]}
        
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                return {
                    "confidence": confidence,
                    "predicted_class": predicted_class,
                    "probabilities": probabilities[0].cpu().numpy().tolist()
                }
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return {"confidence": 0.5, "predicted_class": 0, "probabilities": [0.5, 0.5]}

class DamageClassifier(CarConditionClassifier):
    """Классификатор повреждений (битый/не битый)"""
    
    def __init__(self, model_path: str = "models/damage_classifier.pth"):
        super().__init__(model_path, num_classes=2, model_type="resnet18")
        self.class_names = ["не битый", "битый"]
    
    def predict_damage(self, image_tensor: torch.Tensor) -> Dict:
        """Предсказание наличия повреждений"""
        result = self.predict(image_tensor)
        return {
            "damaged": result["predicted_class"],
            "confidence": result["confidence"],
            "class_name": self.class_names[result["predicted_class"]],
            "probabilities": {
                "не битый": result["probabilities"][0],
                "битый": result["probabilities"][1]
            }
        }

class DirtClassifier(CarConditionClassifier):
    """Классификатор загрязнения (грязный/чистый)"""
    
    def __init__(self, model_path: str = "models/dirt_classifier.pth"):
        super().__init__(model_path, num_classes=2, model_type="resnet18")
        self.class_names = ["чистый", "грязный"]
    
    def predict_dirt(self, image_tensor: torch.Tensor) -> Dict:
        """Предсказание наличия загрязнений"""
        result = self.predict(image_tensor)
        return {
            "dirty": result["predicted_class"],
            "confidence": result["confidence"],
            "class_name": self.class_names[result["predicted_class"]],
            "probabilities": {
                "чистый": result["probabilities"][0],
                "грязный": result["probabilities"][1]
            }
        }

class ScratchClassifier(CarConditionClassifier):
    """Классификатор царапин (есть царапины/нет царапин)"""
    
    def __init__(self, model_path: str = "models/scratch_classifier.pth"):
        super().__init__(model_path, num_classes=2, model_type="resnet18")
        self.class_names = ["нет царапин", "есть царапины"]
    
    def predict_scratch(self, image_tensor: torch.Tensor) -> Dict:
        """Предсказание наличия царапин"""
        result = self.predict(image_tensor)
        return {
            "scratched": result["predicted_class"],
            "confidence": result["confidence"],
            "class_name": self.class_names[result["predicted_class"]],
            "probabilities": {
                "нет царапин": result["probabilities"][0],
                "есть царапины": result["probabilities"][1]
            }
        }

class CarConditionAnalyzer:
    """Главный анализатор состояния автомобиля"""
    
    def __init__(self, 
                 damage_model_path: str = "models/damage_classifier.pth",
                 dirt_model_path: str = "models/dirt_classifier.pth", 
                 scratch_model_path: str = "models/scratch_classifier.pth"):
        """
        Инициализация анализатора со всеми моделями
        
        Args:
            damage_model_path: путь к модели классификации повреждений
            dirt_model_path: путь к модели классификации загрязнений
            scratch_model_path: путь к модели классификации царапин
        """
        self.damage_classifier = DamageClassifier(damage_model_path)
        self.dirt_classifier = DirtClassifier(dirt_model_path)
        self.scratch_classifier = ScratchClassifier(scratch_model_path)
        
        logger.info("Анализатор состояния автомобиля инициализирован")
    
    def analyze_car_condition(self, image_tensor: torch.Tensor) -> Dict:
        """
        Полный анализ состояния автомобиля
        
        Args:
            image_tensor: тензор изображения автомобиля
            
        Returns:
            Словарь с результатами анализа всех параметров
        """
        try:
            # Получаем предсказания от всех моделей
            damage_result = self.damage_classifier.predict_damage(image_tensor)
            dirt_result = self.dirt_classifier.predict_dirt(image_tensor)
            scratch_result = self.scratch_classifier.predict_scratch(image_tensor)
            
            # Формируем итоговый результат в требуемом формате
            result = {
                "битый": damage_result["damaged"],
                "грязный": dirt_result["dirty"],
                "царапины": scratch_result["scratched"],
                "detailed_analysis": {
                    "damage": {
                        "status": damage_result["class_name"],
                        "confidence": round(damage_result["confidence"], 3),
                        "probabilities": damage_result["probabilities"]
                    },
                    "dirt": {
                        "status": dirt_result["class_name"],
                        "confidence": round(dirt_result["confidence"], 3),
                        "probabilities": dirt_result["probabilities"]
                    },
                    "scratch": {
                        "status": scratch_result["class_name"],
                        "confidence": round(scratch_result["confidence"], 3),
                        "probabilities": scratch_result["probabilities"]
                    }
                },
                "overall_condition": self._get_overall_condition(
                    damage_result["damaged"],
                    dirt_result["dirty"],
                    scratch_result["scratched"]
                ),
                "analysis_successful": True,
                "message": "Анализ состояния автомобиля выполнен успешно"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа состояния автомобиля: {e}")
            return {
                "битый": 0,
                "грязный": 0,
                "царапины": 0,
                "detailed_analysis": None,
                "overall_condition": "unknown",
                "analysis_successful": False,
                "message": f"Ошибка анализа: {str(e)}"
            }
    
    def _get_overall_condition(self, damaged: int, dirty: int, scratched: int) -> str:
        """Определение общего состояния автомобиля"""
        conditions = []
        
        if damaged:
            conditions.append("поврежден")
        if dirty:
            conditions.append("загрязнен")
        if scratched:
            conditions.append("поцарапан")
        
        if not conditions:
            return "отличное состояние"
        elif len(conditions) == 1:
            return f"хорошее состояние ({conditions[0]})"
        elif len(conditions) == 2:
            return f"удовлетворительное состояние ({', '.join(conditions)})"
        else:
            return f"плохое состояние ({', '.join(conditions)})"

def create_models_directory():
    """Создание директории для моделей если она не существует"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Создана директория для моделей: {models_dir}")
    
    return models_dir

def save_dummy_models():
    """Создание демонстрационных моделей для тестирования"""
    models_dir = create_models_directory()
    
    model_configs = [
        ("damage_classifier.pth", 2),
        ("dirt_classifier.pth", 2), 
        ("scratch_classifier.pth", 2)
    ]
    
    for model_name, num_classes in model_configs:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            # Создаем простую модель ResNet18
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # Сохраняем модель
            torch.save(model.state_dict(), model_path)
            logger.info(f"Создана демонстрационная модель: {model_path}")

# Автоматически создаем демонстрационные модели при импорте модуля
if __name__ == "__main__":
    save_dummy_models()
else:
    # Создаем модели при импорте, если их нет
    try:
        save_dummy_models()
    except Exception as e:
        logger.warning(f"Не удалось создать демонстрационные модели: {e}")