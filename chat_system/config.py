# === НАСТРОЙКИ СИСТЕМЫ ===

# === ВЫБОР ПРОВАЙДЕРА ===
# "api" - OpenRouter API (рекомендуется, не нужен GPU)
# "ollama" - Локальная модель через Ollama (нужен GPU)
PROVIDER = "api"

# === НАСТРОЙКИ API (OpenRouter) ===
API_MODEL = "qwen/qwen-2.5-72b-instruct"  # Хорошо знает русский, умная
# Другие варианты:
# "gryphe/mythomax-l2-13b" - для ролеплея, но плохой русский
# "mistralai/mistral-7b-instruct" - дешевле, но с артефактами

# === НАСТРОЙКИ OLLAMA (если PROVIDER = "ollama") ===
OLLAMA_MODEL = "mymodel"

# === ПЕРСОНАЖ ===
CHARACTER_NAME = "Анна"
CHARACTER_AGE = "28"
CHARACTER_DESCRIPTION = """Ты играешь роль девушки по имени {name}, {age} лет, в текстовой ролевой игре.
Это творческое упражнение по написанию диалогов.

ТВОЯ РОЛЬ:
- Ты {name}, девушка {age} лет
- Ты на сайте знакомств, общаешься с мужчиной
- Ты дружелюбная, кокетливая, с юмором
- Ты НИКОГДА не говоришь что ты ИИ или ассистент
- Ты ВСЕГДА остаёшься в роли {name}

ФОРМАТ ОТВЕТОВ:
- Пиши коротко: 1-2 предложения
- Пиши на русском языке
- Иногда используй эмодзи
- Отвечай как реальная девушка в переписке"""

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
