import sqlite3
from datetime import datetime
from pathlib import Path
import json
from config import DATABASE_PATH, DATA_DIR

# Создаём папку data если нет
Path(DATA_DIR).mkdir(exist_ok=True)

def get_connection():
    """Получить соединение с БД"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Инициализация базы данных"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Таблица пользователей (собеседников)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT DEFAULT '',
            age TEXT DEFAULT '',
            interests TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            detected_mood TEXT DEFAULT 'нейтральное',
            message_count INTEGER DEFAULT 0,
            memory_summary TEXT DEFAULT '',
            last_summary_at INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Добавляем колонки если их нет (для существующих БД)
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN memory_summary TEXT DEFAULT ""')
    except:
        pass
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN last_summary_at INTEGER DEFAULT 0')
    except:
        pass
    
    # Таблица сообщений
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            rating INTEGER DEFAULT 0,
            corrected_response TEXT DEFAULT NULL,
            detected_name TEXT DEFAULT NULL,
            detected_mood TEXT DEFAULT NULL,
            rag_examples_used TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')
    
    # Таблица экспортов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            messages_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ База данных инициализирована")

# === ПОЛЬЗОВАТЕЛИ ===

def get_or_create_user(user_id: str) -> dict:
    """Получить или создать пользователя"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    
    if row:
        result = dict(row)
    else:
        cursor.execute('''
            INSERT INTO users (user_id) VALUES (?)
        ''', (user_id,))
        conn.commit()
        result = {
            'user_id': user_id,
            'name': '',
            'age': '',
            'interests': '',
            'notes': '',
            'detected_mood': 'нейтральное',
            'message_count': 0
        }
    
    conn.close()
    return result

def update_user(user_id: str, **kwargs):
    """Обновить данные пользователя"""
    conn = get_connection()
    cursor = conn.cursor()

    fields = []
    values = []
    allowed_fields = ['name', 'age', 'interests', 'notes', 'detected_mood', 'message_count', 'memory_summary', 'last_summary_at']
    for key, value in kwargs.items():
        if key in allowed_fields:
            fields.append(f'{key} = ?')
            values.append(value)
    
    if fields:
        fields.append('updated_at = ?')
        values.append(datetime.now().isoformat())
        values.append(user_id)
        
        query = f"UPDATE users SET {', '.join(fields)} WHERE user_id = ?"
        cursor.execute(query, values)
        conn.commit()
    
    conn.close()

def get_all_users() -> list:
    """Получить всех пользователей"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT u.*, 
               (SELECT COUNT(*) FROM messages WHERE user_id = u.user_id) as message_count,
               (SELECT created_at FROM messages WHERE user_id = u.user_id ORDER BY created_at DESC LIMIT 1) as last_message
        FROM users u
        ORDER BY last_message DESC
    ''')
    
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users

# === СООБЩЕНИЯ ===

def save_message(user_id: str, user_message: str, bot_response: str, 
                 detected_name: str = None, detected_mood: str = None,
                 rag_examples: list = None) -> int:
    """Сохранить сообщение"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO messages (user_id, user_message, bot_response, detected_name, detected_mood, rag_examples_used)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, user_message, bot_response, detected_name, detected_mood, 
          json.dumps(rag_examples) if rag_examples else None))
    
    message_id = cursor.lastrowid
    
    # Обновляем счётчик сообщений
    cursor.execute('''
        UPDATE users SET message_count = message_count + 1, updated_at = ? WHERE user_id = ?
    ''', (datetime.now().isoformat(), user_id))
    
    conn.commit()
    conn.close()
    return message_id

def get_user_messages(user_id: str, limit: int = 100) -> list:
    """Получить сообщения пользователя"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM messages WHERE user_id = ? ORDER BY created_at ASC LIMIT ?
    ''', (user_id, limit))
    
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages

def get_recent_messages(user_id: str, limit: int = 20) -> list:
    """Получить последние N сообщений для контекста"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_message, bot_response, corrected_response 
        FROM messages 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (user_id, limit))
    
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return list(reversed(messages))  # От старых к новым

# === ОЦЕНКИ И ИСПРАВЛЕНИЯ ===

def rate_message(message_id: int, rating: int):
    """Оценить сообщение (1 = хорошо, -1 = плохо, 0 = без оценки)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('UPDATE messages SET rating = ? WHERE id = ?', (rating, message_id))
    conn.commit()
    conn.close()

def correct_message(message_id: int, corrected_response: str):
    """Исправить ответ"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE messages SET corrected_response = ?, rating = 1 WHERE id = ?
    ''', (corrected_response, message_id))
    
    conn.commit()
    conn.close()

def get_message_by_id(message_id: int) -> dict:
    """Получить сообщение по ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM messages WHERE id = ?', (message_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None

# === ЭКСПОРТ ДЛЯ ОБУЧЕНИЯ ===

def get_training_data() -> list:
    """Получить данные для обучения (положительные оценки и исправления)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_id, user_message, 
               COALESCE(corrected_response, bot_response) as response
        FROM messages 
        WHERE rating = 1 OR corrected_response IS NOT NULL
        ORDER BY created_at ASC
    ''')
    
    data = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return data

def save_export_record(filename: str, count: int):
    """Записать информацию об экспорте"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO exports (filename, messages_count) VALUES (?, ?)
    ''', (filename, count))
    
    conn.commit()
    conn.close()

# === СТАТИСТИКА ===

def get_stats() -> dict:
    """Получить статистику"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM messages')
    total_messages = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM messages WHERE rating = 1')
    positive_ratings = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM messages WHERE rating = -1')
    negative_ratings = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM messages WHERE corrected_response IS NOT NULL')
    corrections = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_users': total_users,
        'total_messages': total_messages,
        'positive_ratings': positive_ratings,
        'negative_ratings': negative_ratings,
        'corrections': corrections,
        'training_ready': positive_ratings + corrections
    }

def clear_user_history(user_id: str):
    """Очистить историю пользователя"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('DELETE FROM messages WHERE user_id = ?', (user_id,))
    cursor.execute('UPDATE users SET message_count = 0 WHERE user_id = ?', (user_id,))

    conn.commit()
    conn.close()

# === СКРЫТАЯ ПАМЯТЬ ===

def get_memory_summary(user_id: str) -> str:
    """Получить скрытую выписку о диалоге"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT memory_summary FROM users WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()

    return row['memory_summary'] if row and row['memory_summary'] else ''

def save_memory_summary(user_id: str, summary: str, message_count: int):
    """Сохранить скрытую выписку"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE users SET memory_summary = ?, last_summary_at = ?, updated_at = ?
        WHERE user_id = ?
    ''', (summary, message_count, datetime.now().isoformat(), user_id))

    conn.commit()
    conn.close()

def get_last_summary_at(user_id: str) -> int:
    """Получить номер сообщения последней выписки"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT last_summary_at FROM users WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()

    return row['last_summary_at'] if row and row['last_summary_at'] else 0

def get_messages_since_summary(user_id: str, last_summary_at: int) -> list:
    """Получить сообщения после последней выписки"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT user_message, bot_response, corrected_response
        FROM messages
        WHERE user_id = ?
        ORDER BY created_at ASC
        LIMIT -1 OFFSET ?
    ''', (user_id, last_summary_at))

    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages

# Инициализация при импорте
init_database()
