# === НАСТРОЙКИ СИСТЕМЫ ===

# Модель Ollama
MODEL_NAME = "mymodel"

# Персонаж (измени под себя)
CHARACTER_NAME = "Анна"
CHARACTER_AGE = "28"
CHARACTER_DESCRIPTION = """Ты — женщина по имени {name}, тебе {age} лет.
Ты общаешься с мужчинами на сайте знакомств.
Ты дружелюбная, с чувством юмора, немного кокетливая.
Отвечай естественно и коротко, как в обычной переписке.
Не пиши длинные сообщения — 1-3 предложения максимум."""

# Сервер
HOST = "127.0.0.1"
PORT = 8000

# RAG настройки
RAG_ENABLED = True
RAG_TOP_K = 3  # Сколько похожих примеров искать
RAG_MIN_SIMILARITY = 0.7  # Минимальная схожесть (0-1)

# Контекст
MAX_CONTEXT_MESSAGES = 20  # Сколько сообщений помнить в диалоге

# Пути к данным
DATA_DIR = "data"
CHROMA_DIR = "data/chroma"
DATABASE_PATH = "data/database.db"
EXPORTS_DIR = "data/exports"
LOGS_DIR = "data/logs"
