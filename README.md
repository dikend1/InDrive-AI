# 🚗 Car Condition Analyzer API

Система анализа состояния автомобилей с использованием машинного обучения для определения повреждений, загрязнений и царапин по фотографиям.

## 📋 Описание проекта

Car Condition Analyzer API - это веб-сервис на базе FastAPI, который анализирует состояние автомобилей по загружаемым изображениям. Система определяет три основных параметра:

- **Битый/Не битый** (0 или 1)
- **Грязный/Чистый** (0 или 1)  
- **Царапины/Нет царапин** (0 или 1)

### ✨ Основные функции

- ✅ Автоматическая детекция автомобилей на изображении
- ✅ Классификация повреждений (вмятины, деформации)
- ✅ Определение степени загрязнения
- ✅ Обнаружение царапин на кузове
- ✅ JSON API с подробнымиg результатами
- ✅ Валидация загружаемых изображений
- ✅ Подробная документация API (Swagger UI)

## 🛠️ Технологии

- **Python 3.11+**
- **FastAPI** - веб-фреймворк для API
- **PyTorch** - машинное обучение
- **YOLO (Ultralytics)** - детекция объектов
- **OpenCV** - обработка изображений
- **PIL (Pillow)** - работа с изображениями
- **Uvicorn** - ASGI сервер

## 📁 Структура проекта

```
inDrive/
├── main.py                    # Основной FastAPI сервис
├── model.py                   # Модели машинного обучения
├── utils.py                   # Утилиты для обработки изображений
├── test_api.py                # Скрипт для тестирования API
├── start_server.sh            # Скрипт запуска сервера
├── test_web_interface.html    # Веб-интерфейс для тестирования
├── requirements.txt           # Зависимости Python
├── README.md                 # Документация проекта
├── data/                     # Датасет 1 (царапины)
├── data1/                    # Датасет 2 (ржавчина и царапины)
├── data2/                    # Датасет 3 (вмятины и грязь)
├── models/                   # Обученные модели ML (создается автоматически)
└── venv/                     # Виртуальное окружение Python
```

## 🚀 Быстрый старт

### 1. Клонирование и настройка

```bash
# Переход в директорию проекта
cd /Users/diasmaksatov/inDrive

# Активация виртуального окружения (если не активировано)
source venv/bin/activate
```

### 2. Установка зависимостей

```bash
# Установка дополнительных пакетов (если нужно)
pip install fastapi uvicorn python-multipart
```

### 3. Запуск сервера

```bash
# Простой запуск через скрипт (рекомендуется)
./start_server.sh

# Или запуск в режиме разработки
python main.py

# Или через uvicorn напрямую
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Проверка работы

Откройте браузер и перейдите по адресам:

- **API документация**: http://localhost:8001/docs (или другой порт, который покажет скрипт)
- **Главная страница**: http://localhost:8001
- **Проверка здоровья**: http://localhost:8001/health
- **Веб-интерфейс для тестирования**: откройте файл `test_web_interface.html` в браузере

### 5. Тестирование системы

```bash
# Запуск автоматических тестов API
python test_api.py

# Или используйте веб-интерфейс
# Откройте test_web_interface.html в браузере
```

## 📚 API документация

### Основные эндпоинты

#### `GET /` - Информация о API
```json
{
  "message": "Car Condition Analyzer API",
  "version": "1.0.0",
  "description": "API для анализа состояния автомобилей",
  "endpoints": {
    "analyze": "/analyze - анализ изображения автомобиля",
    "health": "/health - проверка состояния API",
    "docs": "/docs - документация API"
  }
}
```

#### `GET /health` - Проверка состояния системы
```json
{
  "status": "healthy",
  "image_processor": true,
  "condition_analyzer": true,
  "models_loaded": {
    "damage_classifier": true,
    "dirt_classifier": true,
    "scratch_classifier": true
  }
}
```

#### `POST /analyze` - Анализ изображения автомобиля

**Параметры:**
- `file` (обязательный): изображение автомобиля (JPEG, PNG)

**Пример успешного ответа:**
```json
{
  "битый": 0,
  "грязный": 1,
  "царапины": 0,
  "detailed_analysis": {
    "damage": {
      "status": "не битый",
      "confidence": 0.892,
      "probabilities": {
        "не битый": 0.892,
        "битый": 0.108
      }
    },
    "dirt": {
      "status": "грязный",
      "confidence": 0.756,
      "probabilities": {
        "чистый": 0.244,
        "грязный": 0.756
      }
    },
    "scratch": {
      "status": "нет царапин",
      "confidence": 0.943,
      "probabilities": {
        "нет царапин": 0.943,
        "есть царапины": 0.057
      }
    }
  },
  "overall_condition": "хорошее состояние (загрязнен)",
  "analysis_successful": true,
  "message": "Анализ состояния автомобиля выполнен успешно",
  "car_detected": true,
  "car_detection_confidence": 0.89,
  "analysis_performed": true
}
```

**Пример ответа когда автомобиль не найден:**
```json
{
  "битый": 0,
  "грязный": 0,
  "царапины": 0,
  "message": "На изображении не обнаружен автомобиль",
  "car_detected": false,
  "analysis_performed": false
}
```

#### `POST /analyze-detailed` - Подробный анализ

Возвращает расширенную информацию включая детекцию автомобилей и метаданные изображения.

#### `GET /models/status` - Статус моделей

```json
{
  "damage_classifier": {
    "loaded": true,
    "model_path": "models/damage_classifier.pth",
    "classes": ["не битый", "битый"]
  },
  "dirt_classifier": {
    "loaded": true,
    "model_path": "models/dirt_classifier.pth",
    "classes": ["чистый", "грязный"]
  },
  "scratch_classifier": {
    "loaded": true,
    "model_path": "models/scratch_classifier.pth",
    "classes": ["нет царапин", "есть царапины"]
  }
}
```

## 🔧 Примеры использования

### Python (requests)

```python
import requests

# Анализ изображения
url = "http://localhost:8000/analyze"
files = {"file": open("car_image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Битый: {result['битый']}")
print(f"Грязный: {result['грязный']}")  
print(f"Царапины: {result['царапины']}")
```

### cURL

```bash
# Анализ изображения
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@car_image.jpg"

# Проверка состояния API
curl -X GET "http://localhost:8000/health"
```

### JavaScript (fetch)

```javascript
async function analyzeCarImage(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    try {
        const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        console.log('Битый:', result.битый);
        console.log('Грязный:', result.грязный);
        console.log('Царапины:', result.царапины);
        
        return result;
    } catch (error) {
        console.error('Ошибка анализа:', error);
    }
}

// Использование с input файла
document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        analyzeCarImage(file);
    }
});
```

## ⚙️ Конфигурация

### Ограничения файлов

- **Максимальный размер**: 10 MB
- **Форматы**: JPEG, PNG, JPG
- **Минимальные размеры**: 100x100 пикселей
- **Максимальные размеры**: 4096x4096 пикселей

### Настройки детекции

- **Порог уверенности детекции автомобиля**: 0.5
- **Модель детекции**: YOLOv8n
- **Класс автомобиля в COCO**: 2 (car)

## 🐛 Диагностика и решение проблем

### Частые ошибки

1. **"Система анализа не инициализирована" (503)**
   - Проверьте, что все модули загружены корректно
   - Перезапустите сервер

2. **"Неподдерживаемый формат изображения" (400)**
   - Используйте JPEG или PNG
   - Проверьте, что файл не поврежден

3. **"Файл слишком большой" (400)**
   - Уменьшите размер изображения до 10 MB

4. **"На изображении не обнаружен автомобиль"**
   - Убедитесь, что автомобиль четко виден
   - Попробуйте изображение с лучшим качеством
   - Проверьте освещение на фото

### Логи и отладка

Логи сервера выводятся в консоль. Для подробного логирования:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Информация о моделях

Система использует три специализированные модели на базе ResNet18:

1. **Модель повреждений** (`damage_classifier.pth`)
   - Обученна на датасетах вмятин и деформаций
   - Точность: определяется при обучении

2. **Модель загрязнений** (`dirt_classifier.pth`)
   - Определяет степень загрязнения кузова
   - Включает пыль, грязь, пятна

3. **Модель царапин** (`scratch_classifier.pth`)
   - Обнаруживает царапины различной глубины
   - Анализирует повреждения лакокрасочного покрытия

### Обучение собственных моделей

Для обучения собственных моделей используйте предоставленные датасеты:

```python
from model import CarConditionClassifier

# Создание и обучение модели
classifier = CarConditionClassifier(
    model_path="models/custom_model.pth",
    num_classes=2,
    model_type="resnet18"
)

# Добавьте код обучения здесь
```

## 🔒 Безопасность

- API не требует аутентификации (для демонстрации)
- Загружаемые файлы валидируются по типу и размеру
- Обработка изображений происходит в памяти без сохранения на диск
- CORS настроен для всех источников (только для разработки)

## 📈 Производительность

- **Время обработки**: 1-3 секунды на изображение
- **Поддерживаемая нагрузка**: зависит от железа
- **Использование GPU**: автоматически при наличии CUDA
- **Память**: ~1-2 GB для загруженных моделей

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Создайте Pull Request

## 📄 Лицензия

Датасеты используются под лицензией CC BY 4.0.
Код проекта доступен для использования в образовательных целях.

## 📞 Поддержка

При возникновении проблем:

1. Проверьте `/health` эндпоинт
2. Изучите логи сервера
3. Убедитесь в корректности формата изображения
4. Проверьте наличие автомобиля на изображении

---

**Разработано для анализа состояния автомобилей с использованием современных технологий машинного обучения.**# InDrive-AI
