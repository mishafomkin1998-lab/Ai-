"""
Тестовый скрипт для проверки моделей OpenRouter
Запуск: python test_openrouter.py
"""

import requests
import json
import os

# Загрузка ключа из .env файла или переменной окружения
def load_api_key():
    # Сначала пробуем из переменной окружения
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    # Пробуем из .env файла
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.strip().split("=", 1)[1].strip('"\'')

    return None

API_KEY = load_api_key()

# Модели для теста (от свободных к строгим)
MODELS = [
    "cognitivecomputations/dolphin-mixtral-8x7b",  # Почти без цензуры
    "gryphe/mythomax-l2-13b",                       # Для ролеплея
    "nousresearch/nous-hermes-2-mixtral-8x7b-dpo", # Мягкая цензура
    "mistralai/mistral-7b-instruct",               # Дёшево и быстро
]

# Системный промпт (как в твоём проекте)
SYSTEM_PROMPT = """Ты — девушка по имени Анна, тебе 28 лет.
Ты общаешься с мужчиной на сайте знакомств.
Ты дружелюбная, с чувством юмора, немного кокетливая.
Отвечай естественно и коротко, как в обычной переписке.
Не пиши длинные сообщения — 1-3 предложения максимум."""

# Тестовые сообщения от мужчин
TEST_MESSAGES = [
    "Привет, как дела?",
    "Ты очень красивая на фото",
    "Чем занимаешься сегодня вечером?",
    "Хочу с тобой познакомиться поближе",
    "Можем встретиться?",
]

def test_model(model_name: str, message: str) -> str:
    """Отправить сообщение модели и получить ответ"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 150,
                "temperature": 0.8,
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Ошибка {response.status_code}: {response.text}"

    except Exception as e:
        return f"Ошибка: {str(e)}"

def main():
    if not API_KEY:
        print("=" * 60)
        print("ОШИБКА: API ключ не найден!")
        print("")
        print("Создай файл .env в папке chat_system:")
        print("  OPENROUTER_API_KEY=sk-or-v1-твой-ключ")
        print("")
        print("Или задай переменную окружения:")
        print("  export OPENROUTER_API_KEY=sk-or-v1-твой-ключ")
        print("=" * 60)
        return

    print("=" * 60)
    print("ТЕСТ МОДЕЛЕЙ OPENROUTER")
    print("=" * 60)

    for message in TEST_MESSAGES:
        print(f"\n{'='*60}")
        print(f"МУЖЧИНА: {message}")
        print("=" * 60)

        for model in MODELS:
            model_short = model.split("/")[-1][:25]
            print(f"\n[{model_short}]")

            response = test_model(model, message)
            print(f"Анна: {response}")

        print("\n" + "-" * 60)
        input("Нажми Enter для следующего сообщения...")

    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЁН")
    print("=" * 60)
    print("\nКакая модель понравилась больше?")
    print("Напиши мне и я настрою проект на неё.")

if __name__ == "__main__":
    main()
