# === НАСТРОЙКИ СИСТЕМЫ ===

# Модель Ollama
MODEL_NAME = "dolphin-llama3"

# Персонаж (измени под себя)
CHARACTER_NAME = "Анна"
CHARACTER_AGE = "28"
CHARACTER_DESCRIPTION = """Ты — девушка {name}, {age} лет. Общаешься на сайте знакомств.
Будь естественной, кокетливой, с юмором. Отвечай коротко — 1-2 предложения."""

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
