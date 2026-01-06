# === НАСТРОЙКИ СИСТЕМЫ ===

# Модель Ollama
MODEL_NAME = "mymodel"  # Текущая активная модель

# Доступные модели для переключения
AVAILABLE_MODELS = [
    {"name": "mymodel", "description": "Обученная модель (ваша)"},
    {"name": "qwen2.5:7b-instruct", "description": "Qwen 2.5 7B - хороший русский"},
    {"name": "mistral:7b-instruct", "description": "Mistral 7B - стабильные ответы"},
    {"name": "llama3.1:8b-instruct", "description": "Llama 3.1 8B - базовая"},
]

# Персонаж — ПУСТО для обученной модели (она уже знает как отвечать)
CHARACTER_NAME = ""
CHARACTER_AGE = ""
CHARACTER_DESCRIPTION = ""

# Сервер
HOST = "127.0.0.1"
PORT = 8000

# RAG настройки
RAG_ENABLED = False  # Отключено для стабильности
RAG_TOP_K = 3
RAG_MIN_SIMILARITY = 0.7

# Контекст
MAX_CONTEXT_MESSAGES = 15  # Больше истории для лучшей памяти

# Пути к данным
DATA_DIR = "data"
CHROMA_DIR = "data/chroma"
DATABASE_PATH = "data/database.db"
EXPORTS_DIR = "data/exports"
LOGS_DIR = "data/logs"
PROMPT_FILE = "data/custom_prompt.json"
