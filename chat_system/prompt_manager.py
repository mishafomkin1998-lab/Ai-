"""
Управление кастомными промптами
"""
import json
from pathlib import Path
from config import PROMPT_FILE, CHARACTER_NAME, CHARACTER_AGE, CHARACTER_DESCRIPTION

# Дефолтный промпт
DEFAULT_PROMPT = {
    "character_name": CHARACTER_NAME,
    "character_age": CHARACTER_AGE,
    "character_description": CHARACTER_DESCRIPTION,
    "additional_instructions": "",
    "style_instructions": "",
    "forbidden_topics": "",
    "allowed_topics": ""
}

def _ensure_file():
    """Создать файл если не существует"""
    path = Path(PROMPT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PROMPT, f, ensure_ascii=False, indent=2)

def get_prompt() -> dict:
    """Получить текущий промпт"""
    _ensure_file()
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Добавляем недостающие поля
            for key, value in DEFAULT_PROMPT.items():
                if key not in data:
                    data[key] = value
            return data
    except Exception as e:
        print(f"Ошибка чтения промпта: {e}")
        return DEFAULT_PROMPT.copy()

def save_prompt(data: dict) -> bool:
    """Сохранить промпт"""
    _ensure_file()
    try:
        # Валидация
        required = ["character_name", "character_description"]
        for field in required:
            if not data.get(field):
                return False

        with open(PROMPT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Ошибка сохранения промпта: {e}")
        return False

def build_full_prompt() -> str:
    """Построить полный системный промпт"""
    data = get_prompt()

    # Основное описание
    prompt = data["character_description"].format(
        name=data["character_name"],
        age=data.get("character_age", "")
    )

    # Дополнительные инструкции
    if data.get("additional_instructions"):
        prompt += f"\n\n{data['additional_instructions']}"

    # Стиль общения
    if data.get("style_instructions"):
        prompt += f"\n\nСтиль общения: {data['style_instructions']}"

    # Разрешённые темы
    if data.get("allowed_topics"):
        prompt += f"\n\nТы можешь обсуждать: {data['allowed_topics']}"

    # Запрещённые темы (если есть)
    if data.get("forbidden_topics"):
        prompt += f"\n\nИзбегай тем: {data['forbidden_topics']}"

    return prompt

def reset_to_default() -> bool:
    """Сбросить на дефолтный промпт"""
    try:
        with open(PROMPT_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PROMPT, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False
