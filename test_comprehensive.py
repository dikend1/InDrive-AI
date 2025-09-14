#!/usr/bin/env python3
"""
Тест логики анализа автомобиля - проверяем все аспекты
"""

import torch
from model import CarConditionAnalyzer

def test_all_aspects():
    """Тестируем все три аспекта для одного изображения"""
    print("🔍 Тестирование всех аспектов анализа автомобиля")
    print("=" * 50)
    
    # Создаем тестовое изображение (имитация чистого салона)
    test_image = torch.randn(3, 224, 224)
    print(f"📸 Тестовое изображение: {test_image.shape}")
    
    # Создаем анализаторы для каждого аспекта
    aspects = {
        "битый": CarConditionAnalyzer("models/damage_classifier.pth", "damage"),
        "грязный": CarConditionAnalyzer("models/dirt_classifier.pth", "dirt"), 
        "царапины": CarConditionAnalyzer("models/scratch_classifier.pth", "scratch")
    }
    
    results = {}
    
    for aspect_name, analyzer in aspects.items():
        result = analyzer.predict(test_image)
        results[aspect_name] = result['predicted_class']
        
        print(f"\n{aspect_name.upper()}:")
        print(f"  Результат: {result['predicted_class']} ({'есть проблема' if result['predicted_class'] == 1 else 'все хорошо'})")
        print(f"  Уверенность: {result['confidence']:.3f}")
        print(f"  Вероятности: {[f'{p:.3f}' for p in result['probabilities']]}")
    
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
    print(f"битый: {results['битый']}")
    print(f"грязный: {results['грязный']}")
    print(f"царапины: {results['царапины']}")
    
    # Проверяем ожидаемый результат
    all_clean = all(value == 0 for value in results.values())
    
    if all_clean:
        print("✅ ОТЛИЧНО! Все аспекты показывают 'чистое' состояние (0,0,0)")
    else:
        print("❌ ПРОБЛЕМА! Найдены неожиданные положительные результаты:")
        for aspect, value in results.items():
            if value == 1:
                print(f"   - {aspect}: {value} (должно быть 0)")
    
    return results

def test_multiple_images():
    """Тестируем несколько разных изображений"""
    print("\n\n🔍 Тестирование множественных изображений")
    print("=" * 50)
    
    analyzer = CarConditionAnalyzer("models/scratch_classifier.pth", "scratch")
    
    clean_count = 0
    total_count = 10
    
    for i in range(total_count):
        # Создаем разные тестовые изображения
        test_image = torch.randn(3, 224, 224) * (i + 1) * 0.1  # разная вариативность
        result = analyzer.predict(test_image)
        
        if result['predicted_class'] == 0:
            clean_count += 1
            status = "✅ чистое"
        else:
            status = "⚠️  царапины"
            
        print(f"Изображение {i+1:2d}: {status} (уверенность: {result['confidence']:.3f})")
    
    print(f"\n📈 Статистика царапин: {clean_count}/{total_count} изображений определены как чистые")
    
    if clean_count >= total_count * 0.95:  # 95% должны быть чистыми
        print("✅ Система работает корректно - большинство изображений чистые")
    else:
        print("❌ Проблема! Слишком много ложных срабатываний")
    
    return clean_count / total_count

if __name__ == "__main__":
    print("🚗 Тест системы анализа состояния автомобиля")
    print("🎯 Цель: убедиться что чистые изображения дают результат 0,0,0")
    
    # Тест 1: все аспекты
    main_results = test_all_aspects()
    
    # Тест 2: множественные изображения
    clean_ratio = test_multiple_images()
    
    print(f"\n🏁 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
    print(f"   Основной тест: {'✅ ПРОЙДЕН' if all(v == 0 for v in main_results.values()) else '❌ НЕ ПРОЙДЕН'}")
    print(f"   Статистика: {clean_ratio:.1%} изображений определены как чистые")
    
    if all(v == 0 for v in main_results.values()) and clean_ratio >= 0.95:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система работает корректно.")
    else:
        print("🚨 ЕСТЬ ПРОБЛЕМЫ! Требуется дополнительная отладка.")