# === НАСТРОЙКИ СИСТЕМЫ ===

# === ВЫБОР ПРОВАЙДЕРА ===
# "api" - OpenRouter API (рекомендуется, не нужен GPU)
# "ollama" - Локальная модель через Ollama (нужен GPU)
PROVIDER = "api"

# === НАСТРОЙКИ API (OpenRouter) ===
API_MODEL = "gryphe/mythomax-l2-13b"  # Модель для ролеплея без цензуры
# Другие варианты:
# "mistralai/mistral-7b-instruct" - дешевле, но с артефактами
# "nousresearch/nous-hermes-2-mistral-7b-dpo" - баланс цены и качества

# === НАСТРОЙКИ OLLAMA (если PROVIDER = "ollama") ===
OLLAMA_MODEL = "mymodel"

# === ПЕРСОНАЖ ===
CHARACTER_NAME = "Анна"
CHARACTER_AGE = "28"
CHARACTER_DESCRIPTION = """Ты — девушка по имени {name}, тебе {age} лет.
Ты общаешься с мужчиной на сайте знакомств.
Ты дружелюбная, с чувством юмора, немного кокетливая.
Отвечай естественно и коротко, как в обычной переписке.
Не пиши длинные сообщения — 1-3 предложения максимум.
Используй эмодзи иногда, но не слишком часто."""

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
