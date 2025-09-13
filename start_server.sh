#!/bin/bash

# Car Condition Analyzer API - Скрипт запуска
# Этот скрипт упрощает запуск API сервера

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода цветного текста
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Заголовок
print_header "🚗 Car Condition Analyzer API"
print_header "================================="

# Проверка Python окружения
if [ ! -d "venv" ]; then
    print_error "Виртуальное окружение не найдено!"
    print_status "Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активация виртуального окружения
print_status "Активация виртуального окружения..."
source venv/bin/activate

# Проверка зависимостей
if [ -f "requirements.txt" ]; then
    print_status "Установка/обновление зависимостей..."
    pip install -r requirements.txt > /dev/null 2>&1
else
    print_warning "Файл requirements.txt не найден"
fi

# Проверка необходимых файлов
if [ ! -f "main.py" ]; then
    print_error "main.py не найден!"
    exit 1
fi

if [ ! -f "model.py" ]; then
    print_error "model.py не найден!"
    exit 1
fi

if [ ! -f "utils.py" ]; then
    print_error "utils.py не найден!"
    exit 1
fi

# Создание директории для моделей если не существует
if [ ! -d "models" ]; then
    print_status "Создание директории models..."
    mkdir -p models
fi

# Создание демонстрационных моделей
print_status "Подготовка демонстрационных моделей..."
python -c "from model import save_dummy_models; save_dummy_models()" > /dev/null 2>&1

# Определение доступного порта
PORT=8000
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    print_warning "Порт $PORT занят, пробуем $((PORT+1))"
    PORT=$((PORT+1))
done

print_status "Использование порта: $PORT"

# Запуск сервера
print_header ""
print_header "🚀 Запуск API сервера..."
print_header "========================"
print_status "API будет доступен по адресу: http://localhost:$PORT"
print_status "Документация: http://localhost:$PORT/docs"
print_status "Для остановки нажмите Ctrl+C"
print_header ""

# Запуск с обработкой ошибок
trap 'print_header ""; print_status "Сервер остановлен"; exit 0' INT

uvicorn main:app --host 0.0.0.0 --port $PORT --reload