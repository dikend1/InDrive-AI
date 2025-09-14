#!/usr/bin/env python3
"""
Простой тест упрощенного API анализа состояния автомобилей
Демонстрирует четкие результаты: битый 0/1, грязный 0/1, царапины 0/1
"""

import requests
import json
import os
import time

API_URL = "http://localhost:8004"

def test_health():
    """Проверка состояния системы"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Система готова к работе")
            return True
        else:
            print(f"❌ Система недоступна: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def test_analyze(image_path="data1/train/1200x900_jpg.rf.db09d6fa3553cf1895c5ccdc390ba7ed.jpg"):
    """Тест основного анализа"""
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return
    
    print(f"\n🔍 Анализ изображения: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/analyze", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Анализ выполнен!")
            print(f"📊 Результат: битый={data['битый']}, грязный={data['грязный']}, царапины={data['царапины']}")
            
            # Интерпретация результата
            status = []
            if data['битый'] == 1:
                status.append("БИТЫЙ")
            if data['грязный'] == 1:
                status.append("ГРЯЗНЫЙ") 
            if data['царапины'] == 1:
                status.append("ЕСТЬ ЦАРАПИНЫ")
            
            if not status:
                print("🎉 Автомобиль в отличном состоянии!")
            else:
                print(f"⚠️  Обнаружено: {', '.join(status)}")
                
        else:
            print(f"❌ Ошибка: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def test_analyze_parts(image_path="data1/train/1200x900_jpg.rf.db09d6fa3553cf1895c5ccdc390ba7ed.jpg"):
    """Тест анализа по частям"""
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return
        
    print(f"\n🧩 Анализ по частям: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/analyze-by-parts", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Анализ по частям выполнен!")
            print(f"📊 Общий результат: битый={data['битый']}, грязный={data['грязный']}, царапины={data['царапины']}")
            
            if 'части' in data and data['части']:
                print("📋 Детализация по частям:")
                for part_name, part_result in data['части'].items():
                    problems = []
                    if part_result.get('битый', 0) == 1:
                        problems.append("битый")
                    if part_result.get('грязный', 0) == 1:
                        problems.append("грязный")
                    if part_result.get('царапины', 0) == 1:
                        problems.append("царапины")
                    
                    if problems:
                        print(f"   • {part_name}: {', '.join(problems)}")
                    else:
                        print(f"   • {part_name}: в хорошем состоянии")
            else:
                print("📋 Анализ по частям не содержит детализации")
                
        else:
            print(f"❌ Ошибка: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def main():
    """Главная функция тестирования"""
    print("🚗 Тестирование упрощенного API анализа автомобилей")
    print("=" * 60)
    
    # Проверка системы
    if not test_health():
        print("\n❌ Система не готова. Запустите сервер:")
        print("   python main.py")
        return
    
    # Поиск доступных изображений
    test_images = []
    for folder in ["data/train", "data1/train", "data2/train"]:
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in images[:2]:  # Берем первые 2 изображения
                test_images.append(os.path.join(folder, img))
    
    if not test_images:
        print("❌ Тестовые изображения не найдены")
        return
    
    # Тестирование с найденными изображениями
    for img_path in test_images[:3]:  # Тестируем максимум 3 изображения
        test_analyze(img_path)
        test_analyze_parts(img_path)
        time.sleep(1)  # Небольшая пауза между запросами
    
    print("\n🎯 Тестирование завершено!")
    print("\n📋 Формат результата:")
    print("   битый: 1 (есть повреждения) или 0 (нет повреждений)")
    print("   грязный: 1 (загрязненный) или 0 (чистый)")
    print("   царапины: 1 (есть царапины) или 0 (нет царапин)")

if __name__ == "__main__":
    main()