"""
ИНСТРУМЕНТ ДЛЯ СОЗДАНИЯ КАЧЕСТВЕННОГО ДАТАСЕТА

Вариант Б: Создание нового качественного датасета вместо очистки мусорных данных.

Использование:
    python dataset_builder.py --action extract    # Извлечь лучшие диалоги из БД
    python dataset_builder.py --action validate   # Проверить качество датасета
    python dataset_builder.py --action generate   # Сгенерировать примеры (шаблон)
    python dataset_builder.py --action export     # Экспортировать финальный датасет
"""

import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from config import DATABASE_PATH, EXPORTS_DIR, CHARACTER_DESCRIPTION, CHARACTER_NAME, CHARACTER_AGE

# Создаём папки
Path(EXPORTS_DIR).mkdir(parents=True, exist_ok=True)
CURATED_DIR = Path("data/curated")
CURATED_DIR.mkdir(parents=True, exist_ok=True)


# === КРИТЕРИИ КАЧЕСТВА ===
QUALITY_CRITERIA = {
    "min_response_length": 5,        # Минимум символов в ответе
    "max_response_length": 300,      # Максимум символов (чтобы не были слишком длинные)
    "min_dialog_turns": 4,           # Минимум реплик в диалоге
    "max_dialog_turns": 50,          # Максимум реплик
    "required_patterns": [           # Должны присутствовать хотя бы некоторые паттерны
        "?",                          # Вопросы
        "...",                        # Паузы/интрига
    ],
    "forbidden_patterns": [          # Запрещённые паттерны (признак плохого ответа)
        "я не могу",
        "я искусственный интеллект",
        "я бот",
        "как языковая модель",
        "я не настоящая",
        "извините, но я",
    ],
}


def get_connection():
    """Подключение к БД"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def extract_best_dialogs(min_rating: int = 1, limit: int = 100) -> List[Dict]:
    """
    Извлекает лучшие диалоги из базы данных.
    Критерии: положительные оценки, исправленные ответы, длина диалога.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Получаем все сообщения с хорошими оценками или исправлениями
    cursor.execute('''
        SELECT user_id, user_message, bot_response, corrected_response, rating, created_at
        FROM messages
        WHERE rating >= ? OR corrected_response IS NOT NULL
        ORDER BY user_id, created_at
    ''', (min_rating,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("Нет данных с положительными оценками. Оцените диалоги в интерфейсе.")
        return []

    # Группируем по пользователям
    dialogs_by_user = {}
    for row in rows:
        user_id = row['user_id']
        if user_id not in dialogs_by_user:
            dialogs_by_user[user_id] = []

        response = row['corrected_response'] or row['bot_response']
        dialogs_by_user[user_id].append({
            "user": row['user_message'],
            "assistant": response,
            "rating": row['rating']
        })

    # Фильтруем по качеству
    quality_dialogs = []
    for user_id, messages in dialogs_by_user.items():
        if len(messages) < QUALITY_CRITERIA["min_dialog_turns"]:
            continue
        if len(messages) > QUALITY_CRITERIA["max_dialog_turns"]:
            messages = messages[:QUALITY_CRITERIA["max_dialog_turns"]]

        # Проверяем качество ответов
        valid_messages = []
        for msg in messages:
            response = msg["assistant"]

            # Проверка длины
            if len(response) < QUALITY_CRITERIA["min_response_length"]:
                continue
            if len(response) > QUALITY_CRITERIA["max_response_length"]:
                continue

            # Проверка запрещённых паттернов
            has_forbidden = any(p in response.lower() for p in QUALITY_CRITERIA["forbidden_patterns"])
            if has_forbidden:
                continue

            valid_messages.append(msg)

        if len(valid_messages) >= QUALITY_CRITERIA["min_dialog_turns"]:
            quality_dialogs.append({
                "user_id": user_id,
                "messages": valid_messages,
                "turns": len(valid_messages)
            })

    # Сортируем по количеству реплик (лучшие диалоги - длиннее)
    quality_dialogs.sort(key=lambda x: x["turns"], reverse=True)

    return quality_dialogs[:limit]


def validate_dataset(filepath: str) -> Dict:
    """Проверяет качество датасета"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {
        "total_dialogs": len(data),
        "total_messages": 0,
        "avg_response_length": 0,
        "questions_ratio": 0,
        "issues": []
    }

    total_length = 0
    questions_count = 0

    for i, dialog in enumerate(data):
        conversations = dialog.get("conversations", [])

        for j, msg in enumerate(conversations):
            if msg.get("from") in ["gpt", "assistant"]:
                stats["total_messages"] += 1
                response = msg.get("value", "")
                total_length += len(response)

                if "?" in response:
                    questions_count += 1

                # Проверки
                if len(response) < 5:
                    stats["issues"].append(f"Диалог {i}, сообщение {j}: слишком короткий ответ")

                for pattern in QUALITY_CRITERIA["forbidden_patterns"]:
                    if pattern in response.lower():
                        stats["issues"].append(f"Диалог {i}, сообщение {j}: запрещённый паттерн '{pattern}'")

    if stats["total_messages"] > 0:
        stats["avg_response_length"] = total_length / stats["total_messages"]
        stats["questions_ratio"] = questions_count / stats["total_messages"]

    return stats


def generate_example_dialogs() -> List[Dict]:
    """
    Генерирует шаблоны диалогов для ручного создания датасета.
    ЭТО ШАБЛОНЫ - нужно заполнить реальными примерами!
    """

    system_prompt = CHARACTER_DESCRIPTION.format(name=CHARACTER_NAME, age=CHARACTER_AGE)

    templates = [
        {
            "name": "Знакомство",
            "description": "Начало общения, флирт",
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": "Привет, ты очень красивая. Давай познакомимся?"},
                {"from": "gpt", "value": "Привет! Спасибо, очень приятно... Расскажи о себе?"},
                {"from": "human", "value": "Меня зовут Андрей, я из Москвы"},
                {"from": "gpt", "value": "Андрей, красивое имя. А чем занимаешься в Москве?"},
            ]
        },
        {
            "name": "Углубление отношений",
            "description": "Переход к более личным темам",
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": "Ты сегодня выглядишь потрясающе"},
                {"from": "gpt", "value": "Спасибо, дорогой... Ты всегда знаешь что сказать"},
                {"from": "human", "value": "У тебя красивая фигура"},
                {"from": "gpt", "value": "Стараюсь следить за собой... А ты занимаешься спортом?"},
            ]
        },
        {
            "name": "Удержание",
            "description": "Когда собеседник хочет уйти или молчит",
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": "Мне пора идти"},
                {"from": "gpt", "value": "Уже? Буду скучать... Напишешь позже?"},
                {"from": "human", "value": "Может быть"},
                {"from": "gpt", "value": "Я буду ждать, дорогой..."},
            ]
        },
        {
            "name": "Откровенный разговор",
            "description": "Переход к интимным темам (заполните сами)",
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": "[ЗАПОЛНИТЕ: откровенный комплимент]"},
                {"from": "gpt", "value": "[ЗАПОЛНИТЕ: принятие с лёгким стеснением]"},
                {"from": "human", "value": "[ЗАПОЛНИТЕ: продолжение]"},
                {"from": "gpt", "value": "[ЗАПОЛНИТЕ: ответ с вопросом]"},
            ]
        },
    ]

    return templates


def export_final_dataset(dialogs: List[Dict], filename: str = None) -> str:
    """Экспортирует финальный датасет в формате ShareGPT"""

    system_prompt = CHARACTER_DESCRIPTION.format(name=CHARACTER_NAME, age=CHARACTER_AGE)

    sharegpt_data = []

    for dialog in dialogs:
        conversations = [{"from": "system", "value": system_prompt}]

        messages = dialog.get("messages", dialog.get("conversations", []))

        for msg in messages:
            if isinstance(msg, dict):
                if "user" in msg:
                    conversations.append({"from": "human", "value": msg["user"]})
                    conversations.append({"from": "gpt", "value": msg["assistant"]})
                elif "from" in msg and msg["from"] != "system":
                    conversations.append(msg)

        if len(conversations) > 1:  # Больше чем только system prompt
            sharegpt_data.append({"conversations": conversations})

    # Сохраняем
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_dataset_{timestamp}.json"

    filepath = Path(EXPORTS_DIR) / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

    print(f"Датасет сохранён: {filepath}")
    print(f"Диалогов: {len(sharegpt_data)}")

    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Инструмент для работы с датасетом")
    parser.add_argument("--action", choices=["extract", "validate", "generate", "export"],
                       required=True, help="Действие")
    parser.add_argument("--file", help="Путь к файлу (для validate)")
    parser.add_argument("--limit", type=int, default=100, help="Лимит диалогов (для extract)")

    args = parser.parse_args()

    if args.action == "extract":
        print("=" * 50)
        print("ИЗВЛЕЧЕНИЕ ЛУЧШИХ ДИАЛОГОВ ИЗ БД")
        print("=" * 50)

        dialogs = extract_best_dialogs(limit=args.limit)

        if dialogs:
            print(f"\nНайдено качественных диалогов: {len(dialogs)}")

            # Сохраняем для ручной проверки
            filepath = CURATED_DIR / "extracted_dialogs.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dialogs, f, ensure_ascii=False, indent=2)

            print(f"Сохранено в: {filepath}")
            print("\nСледующий шаг: просмотрите файл и удалите плохие диалоги вручную")
        else:
            print("Качественных диалогов не найдено.")
            print("Используйте интерфейс для оценки диалогов (👍) или запустите --action generate")

    elif args.action == "validate":
        if not args.file:
            print("Укажите файл: --file path/to/dataset.json")
            return

        print("=" * 50)
        print("ВАЛИДАЦИЯ ДАТАСЕТА")
        print("=" * 50)

        stats = validate_dataset(args.file)

        print(f"\nВсего диалогов: {stats['total_dialogs']}")
        print(f"Всего ответов: {stats['total_messages']}")
        print(f"Средняя длина ответа: {stats['avg_response_length']:.1f} символов")
        print(f"Доля ответов с вопросами: {stats['questions_ratio']*100:.1f}%")

        if stats['issues']:
            print(f"\nПроблемы ({len(stats['issues'])}):")
            for issue in stats['issues'][:10]:
                print(f"  - {issue}")
            if len(stats['issues']) > 10:
                print(f"  ... и ещё {len(stats['issues'])-10}")
        else:
            print("\nПроблем не найдено!")

    elif args.action == "generate":
        print("=" * 50)
        print("ГЕНЕРАЦИЯ ШАБЛОНОВ ДИАЛОГОВ")
        print("=" * 50)

        templates = generate_example_dialogs()

        filepath = CURATED_DIR / "dialog_templates.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)

        print(f"\nШаблоны сохранены в: {filepath}")
        print("\nИНСТРУКЦИЯ:")
        print("1. Откройте файл и заполните шаблоны реальными примерами")
        print("2. Создайте 50-100 диалогов разных типов:")
        print("   - Знакомство (20%)")
        print("   - Флирт (30%)")
        print("   - Откровенные темы (30%)")
        print("   - Удержание (20%)")
        print("3. Запустите validate для проверки")
        print("4. Запустите export для создания финального датасета")

    elif args.action == "export":
        print("=" * 50)
        print("ЭКСПОРТ ФИНАЛЬНОГО ДАТАСЕТА")
        print("=" * 50)

        # Ищем курированные диалоги
        curated_file = CURATED_DIR / "extracted_dialogs.json"
        templates_file = CURATED_DIR / "dialog_templates.json"

        all_dialogs = []

        if curated_file.exists():
            with open(curated_file, 'r', encoding='utf-8') as f:
                all_dialogs.extend(json.load(f))
            print(f"Загружено из extracted_dialogs.json: {len(all_dialogs)} диалогов")

        if templates_file.exists():
            with open(templates_file, 'r', encoding='utf-8') as f:
                templates = json.load(f)
                # Фильтруем только заполненные шаблоны
                filled = [t for t in templates if "[ЗАПОЛНИТЕ" not in json.dumps(t)]
                all_dialogs.extend(filled)
                print(f"Загружено из dialog_templates.json: {len(filled)} диалогов")

        if all_dialogs:
            export_final_dataset(all_dialogs)
        else:
            print("Нет данных для экспорта.")
            print("Сначала запустите --action extract или --action generate")


if __name__ == "__main__":
    main()
