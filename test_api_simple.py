#!/usr/bin/env python3
"""
Простой тест API для проверки результатов анализа
"""

import requests
import json
from io import BytesIO
import numpy as np
from PIL import Image

def create_test_image():
    """Создаем простое тестовое изображение (чистый белый фон)"""
    # Создаем простое белое изображение 224x224
    img_array = np.full((224, 224, 3), 255, dtype=np.uint8)  # белое изображение
    img = Image.fromarray(img_array)
    
    # Сохраняем в BytesIO как JPEG
    img_buffer = BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    return img_buffer.getvalue()

def test_api():
    """Тестируем API с чистым изображением"""
    print("🧪 Тест API анализа автомобилей")
    print("=" * 40)
    
    # URL API (предполагаем что сервер запущен на порту 8000)
    url = "http://localhost:8000/analyze"
    
    # Создаем тестовое изображение
    test_image_data = create_test_image()
    
    # Отправляем запрос
    files = {'file': ('test_image.jpg', test_image_data, 'image/jpeg')}
    
    try:
        print("📤 Отправляем запрос к API...")
        response = requests.post(url, files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Запрос успешен!")
            print(f"📊 Результат: {result}")
            
            # Проверяем результат
            битый = result.get('битый', -1)
            грязный = result.get('грязный', -1) 
            царапины = result.get('царапины', -1)
            
            print(f"\n🔍 Анализ результата:")
            print(f"   битый: {битый} {'✅' if битый == 0 else '❌'}")
            print(f"   грязный: {грязный} {'✅' if грязный == 0 else '❌'}")
            print(f"   царапины: {царапины} {'✅' if царапины == 0 else '❌'}")
            
            # Итоговая проверка
            all_clean = битый == 0 and грязный == 0 and царапины == 0
            
            if all_clean:
                print(f"\n🎉 ОТЛИЧНО! Система правильно определила чистое изображение: ({битый},{грязный},{царапины})")
            else:
                print(f"\n🚨 ПРОБЛЕМА! Чистое изображение определено неправильно: ({битый},{грязный},{царапины})")
                print("   Ожидался результат: (0,0,0)")
                
        else:
            print(f"❌ Ошибка API: {response.status_code}")
            print(f"   Ответ: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к API. Убедитесь что сервер запущен на localhost:8000")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def test_multiple_ports():
    """Тестируем API на разных портах"""
    ports = [8000, 8001, 8002, 8003, 8004, 8005]
    
    for port in ports:
        url = f"http://localhost:{port}/analyze"
        
        try:
            # Быстрая проверка доступности
            health_response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if health_response.status_code == 200:
                print(f"🔍 Найден сервер на порту {port}")
                
                # Создаем тестовое изображение
                test_image_data = create_test_image()
                files = {'file': ('test_image.jpg', test_image_data, 'image/jpeg')}
                
                # Отправляем тестовый запрос
                response = requests.post(url, files=files, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"📊 Порт {port} результат: {result}")
                    
                    # Проверяем царапины
                    царапины = result.get('царапины', -1)
                    if царапины == 0:
                        print(f"✅ Порт {port}: царапины = 0 (правильно)")
                    else:
                        print(f"❌ Порт {port}: царапины = {царапины} (неправильно!)")
                    
                break
                
        except:
            continue
    else:
        print("❌ Не найдено активных серверов API")

if __name__ == "__main__":
    print("🚗 Тест системы анализа автомобилей")
    print("🎯 Проверяем что чистые изображения дают (0,0,0)")
    print()
    
    # Попробуем найти активный сервер
    test_multiple_ports()