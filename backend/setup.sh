#!/bin/bash

echo "⚡ Setting up backend environment..."

# Создаем виртуальное окружение в текущей папке
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Активируем виртуальное окружение
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Обновляем pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Устанавливаем зависимости
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Устанавливаем Parrot отдельно
echo "🦜 Installing Parrot paraphraser..."
pip install git+https://github.com/PrithivirajDamodaran/Parrot_Paraphraser.git

# Создаем папку uploads (если нет)
mkdir -p uploads

echo "✅ Setup complete!"
echo "➡️ To start backend, run:"
echo "source venv/bin/activate && uvicorn main:app --reload"