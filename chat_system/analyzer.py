import re
import subprocess
from config import MODEL_NAME

# === ОПРЕДЕЛЕНИЕ ИМЕНИ ===

# Паттерны для извлечения имени
NAME_PATTERNS = [
    r'меня зовут\s+([А-ЯЁа-яё]+)',
    r'я\s+([А-ЯЁ][а-яё]+)(?:\s|,|\.|\!|\?|$)',
    r'это\s+([А-ЯЁ][а-яё]+)(?:\s|,|\.|\!|\?|$)',
    r'([А-ЯЁ][а-яё]+)\s+на связи',
    r'([А-ЯЁ][а-яё]+)\s+здесь',
    r'зови меня\s+([А-ЯЁа-яё]+)',
    r'можешь звать меня\s+([А-ЯЁа-яё]+)',
    r'имя\s+([А-ЯЁа-яё]+)',
    r'my name is\s+([A-Za-zА-ЯЁа-яё]+)',
    r"i'?m\s+([A-Za-zА-ЯЁа-яё]+)",
]

# Имена которые НЕ нужно извлекать (общие слова)
EXCLUDED_WORDS = {
    'привет', 'здравствуй', 'добрый', 'доброе', 'добрая', 'утро', 'день', 'вечер', 'ночь',
    'как', 'что', 'где', 'когда', 'почему', 'зачем', 'кто', 'чем', 'кем',
    'тут', 'там', 'здесь', 'сейчас', 'потом', 'раньше', 'позже',
    'хорошо', 'плохо', 'нормально', 'отлично', 'супер', 'класс',
    'да', 'нет', 'ага', 'угу', 'ладно', 'окей', 'ок',
    'спасибо', 'пожалуйста', 'извини', 'прости',
    'просто', 'очень', 'совсем', 'вообще', 'конечно', 'наверное',
}

def extract_name_regex(text: str) -> str | None:
    """Извлечь имя из текста с помощью regex"""
    text_lower = text.lower()
    
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Проверяем что это не общее слово
            if name.lower() not in EXCLUDED_WORDS and len(name) >= 2:
                return name.capitalize()
    
    return None

def extract_name_llm(text: str) -> str | None:
    """Извлечь имя с помощью LLM (если regex не сработал)"""
    prompt = f'''Извлеки имя человека из этого сообщения. 
Если имя есть — напиши ТОЛЬКО имя (одно слово).
Если имени нет — напиши "НЕТ".

Сообщение: "{text}"

Имя:'''
    
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30
        )
        response = result.stdout.strip()
        
        # Проверяем ответ
        if response and response.upper() != "НЕТ" and len(response) < 20:
            # Берём первое слово
            name = response.split()[0].strip('.,!?:;')
            if name.lower() not in EXCLUDED_WORDS:
                return name.capitalize()
    except:
        pass
    
    return None

def detect_name(text: str) -> str | None:
    """Определить имя из сообщения"""
    # Сначала пробуем regex (быстро)
    name = extract_name_regex(text)
    if name:
        return name
    
    # Если не нашли и сообщение короткое — пробуем LLM
    if len(text) < 100:
        return extract_name_llm(text)
    
    return None

# === ОПРЕДЕЛЕНИЕ НАСТРОЕНИЯ ===

MOOD_KEYWORDS = {
    'позитивное': [
        'хорошо', 'отлично', 'супер', 'класс', 'круто', 'здорово', 'прекрасно', 'замечательно',
        'рад', 'рада', 'счастлив', 'счастлива', 'весело', 'смешно', 'ха', 'хаха', 'хех', 
        '😊', '😄', '😃', '🙂', '😁', '❤️', '💕', '😍', '🥰', '👍', '🔥', '✨',
        'люблю', 'нравится', 'обожаю', 'восторг'
    ],
    'грустное': [
        'плохо', 'грустно', 'печально', 'тоскливо', 'скучно', 'одиноко',
        'устал', 'устала', 'надоело', 'достало', 'депрессия', 'тоска',
        '😢', '😭', '😔', '😞', '💔', '😿',
        'жаль', 'обидно', 'больно'
    ],
    'флирт': [
        'красивая', 'красотка', 'милая', 'сексуальная', 'горячая', 'привлекательная',
        'нравишься', 'хочу тебя', 'скучаю', 'думаю о тебе', 'мечтаю',
        '😏', '😘', '😉', '💋', '🌹', '💐',
        'встретиться', 'увидеться', 'свидание', 'вместе'
    ],
    'агрессия': [
        'дура', 'идиот', 'тупая', 'бесит', 'ненавижу', 'пошла', 'отвали', 'заткнись',
        'злой', 'злая', 'бешеный', 'бешеная', 'достала', 'достал',
        '😠', '😡', '🤬', '💢',
        'блять', 'сука', 'хуй', 'пизд'
    ],
    'заинтересованность': [
        'интересно', 'расскажи', 'хочу знать', 'любопытно', 'а что', 'а как',
        'правда', 'серьёзно', 'реально', 'вау', 'ого', 'ничего себе',
        '🤔', '🧐', '👀', '❓', '❔',
        'почему', 'зачем', 'откуда', 'сколько'
    ]
}

def detect_mood_keywords(text: str) -> str:
    """Определить настроение по ключевым словам"""
    text_lower = text.lower()
    
    scores = {mood: 0 for mood in MOOD_KEYWORDS}
    
    for mood, keywords in MOOD_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[mood] += 1
    
    # Находим максимум
    max_mood = max(scores, key=scores.get)
    
    if scores[max_mood] > 0:
        return max_mood
    
    return 'нейтральное'

def detect_mood_llm(text: str) -> str:
    """Определить настроение с помощью LLM"""
    prompt = f'''Определи настроение этого сообщения. 
Выбери ОДНО из: позитивное, грустное, флирт, агрессия, заинтересованность, нейтральное

Сообщение: "{text}"

Настроение:'''
    
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30
        )
        response = result.stdout.strip().lower()
        
        # Проверяем ответ
        valid_moods = ['позитивное', 'грустное', 'флирт', 'агрессия', 'заинтересованность', 'нейтральное']
        for mood in valid_moods:
            if mood in response:
                return mood
    except:
        pass
    
    return 'нейтральное'

def detect_mood(text: str, use_llm: bool = False) -> str:
    """Определить настроение сообщения"""
    # Сначала по ключевым словам (быстро)
    mood = detect_mood_keywords(text)
    
    # Если нейтральное и включен LLM — пробуем уточнить
    if mood == 'нейтральное' and use_llm and len(text) > 20:
        mood = detect_mood_llm(text)
    
    return mood

# === АНАЛИЗ СООБЩЕНИЯ ===

def analyze_message(text: str, use_llm_for_mood: bool = False) -> dict:
    """Полный анализ сообщения"""
    return {
        'detected_name': detect_name(text),
        'detected_mood': detect_mood(text, use_llm_for_mood)
    }
