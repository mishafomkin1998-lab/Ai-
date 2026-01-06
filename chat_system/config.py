# === НАСТРОЙКИ СИСТЕМЫ ===

# Модель Ollama
MODEL_NAME = "mymodel"  # Обученная модель

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
MAX_CONTEXT_MESSAGES = 10  # Уменьшено для стабильности

# Пути к данным
DATA_DIR = "data"
CHROMA_DIR = "data/chroma"
DATABASE_PATH = "data/database.db"
EXPORTS_DIR = "data/exports"
LOGS_DIR = "data/logs"
PROMPT_FILE = "data/custom_prompt.json"
