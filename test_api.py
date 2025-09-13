#!/usr/bin/env python3
"""
Примеры использования Car Condition Analyzer API

Этот скрипт демонстрирует различные способы взаимодействия с API
для анализа состояния автомобилей.
"""

import requests
import json
import os
from pathlib import Path

# Базовый URL API
API_BASE_URL = "http://localhost:8001"

def test_health_check():
    """Проверка состояния API"""
    print("🏥 Проверка состояния API...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API работает корректно")
            print(f"   Статус: {data['status']}")
            print(f"   Процессор изображений: {'✅' if data['image_processor'] else '❌'}")
            print(f"   Анализатор состояния: {'✅' if data['condition_analyzer'] else '❌'}")
            return True
        else:
            print(f"❌ Ошибка API: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Не удается подключиться к API. Убедитесь, что сервер запущен на порту 8001")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_analyze_image(image_path):
    """Анализ изображения автомобиля"""
    print(f"\n🚗 Анализ изображения: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/analyze", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Анализ выполнен успешно!")
            print(f"   Битый: {data['битый']} ({'Да' if data['битый'] else 'Нет'})")
            print(f"   Грязный: {data['грязный']} ({'Да' if data['грязный'] else 'Нет'})")
            print(f"   Царапины: {data['царапины']} ({'Да' if data['царапины'] else 'Нет'})")
            
            if 'car_detected' in data:
                print(f"   Автомобиль обнаружен: {'✅' if data['car_detected'] else '❌'}")
            
            if 'overall_condition' in data:
                print(f"   Общее состояние: {data['overall_condition']}")
            
            return data
        else:
            print(f"❌ Ошибка анализа: {response.status_code}")
            print(f"   Ответ: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def test_detailed_analyze(image_path):
    """Подробный анализ изображения"""
    print(f"\n🔍 Подробный анализ: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/analyze-detailed", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Подробный анализ выполнен!")
            
            # Основные результаты
            print(f"   Битый: {data['битый']}, Грязный: {data['грязный']}, Царапины: {data['царапины']}")
            
            # Детальный анализ
            if 'detailed_analysis' in data and data['detailed_analysis']:
                print("\n📊 Детальная информация:")
                for category, info in data['detailed_analysis'].items():
                    print(f"   {category.capitalize()}:")
                    print(f"     Статус: {info['status']}")
                    print(f"     Уверенность: {info['confidence']:.1%}")
            
            # Информация о детекции автомобиля
            if 'car_detection' in data:
                car_det = data['car_detection']
                print(f"\n🚙 Детекция автомобиля:")
                print(f"   Обнаружен: {'✅' if car_det['has_car'] else '❌'}")
                print(f"   Количество: {car_det.get('car_count', 0)}")
            
            return data
        else:
            print(f"❌ Ошибка: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def find_test_images():
    """Поиск тестовых изображений в датасетах"""
    base_path = Path.cwd()
    test_images = []
    
    # Поиск в папках датасетов
    for data_dir in ['data', 'data1', 'data2']:
        data_path = base_path / data_dir
        if data_path.exists():
            # Ищем изображения в папке test
            test_dir = data_path / 'test'
            if test_dir.exists():
                for img_file in test_dir.glob('*.jpg'):
                    test_images.append(str(img_file))
                    if len(test_images) >= 3:  # Берем максимум 3 изображения
                        break
    
    return test_images

def test_models_status():
    """Проверка статуса загруженных моделей"""
    print("\n🧠 Статус моделей машинного обучения:")
    try:
        response = requests.get(f"{API_BASE_URL}/models/status")
        if response.status_code == 200:
            data = response.json()
            
            for model_name, info in data.items():
                print(f"   {model_name.replace('_', ' ').title()}:")
                print(f"     Загружена: {'✅' if info['loaded'] else '❌'}")
                print(f"     Путь: {info['model_path']}")
                print(f"     Классы: {info['classes']}")
        else:
            print(f"❌ Ошибка получения статуса моделей: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def main():
    """Главная функция для демонстрации API"""
    print("🚗 Car Condition Analyzer API - Примеры использования")
    print("=" * 60)
    
    # 1. Проверка состояния API
    if not test_health_check():
        print("\n❌ API недоступен. Убедитесь, что сервер запущен:")
        print("   python main.py")
        print("   или")
        print("   uvicorn main:app --host 0.0.0.0 --port 8001")
        return
    
    # 2. Проверка статуса моделей
    test_models_status()
    
    # 3. Поиск тестовых изображений
    test_images = find_test_images()
    
    if not test_images:
        print("\n⚠️  Тестовые изображения не найдены в папках датасетов")
        print("   Убедитесь, что папки data/, data1/, data2/ содержат изображения")
        return
    
    print(f"\n📁 Найдено {len(test_images)} тестовых изображений")
    
    # 4. Анализ изображений
    for i, image_path in enumerate(test_images[:2], 1):  # Анализируем первые 2 изображения
        print(f"\n{'='*20} Тест {i} {'='*20}")
        
        # Обычный анализ
        result = test_analyze_image(image_path)
        
        # Подробный анализ
        if result and result.get('car_detected', False):
            test_detailed_analyze(image_path)
    
    print(f"\n{'='*60}")
    print("✅ Тестирование завершено!")
    print(f"🌐 Документация API: {API_BASE_URL}/docs")
    print(f"🏠 Главная страница: {API_BASE_URL}/")

if __name__ == "__main__":
    main()