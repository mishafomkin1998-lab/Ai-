from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import subprocess
import uvicorn
from pathlib import Path

from config import (
    MODEL_NAME, HOST, PORT, MAX_CONTEXT_MESSAGES,
    CHARACTER_NAME, CHARACTER_AGE, CHARACTER_DESCRIPTION,
    EXPORTS_DIR
)
import database as db
import analyzer
import rag
import training
import prompt_manager

app = FastAPI(title="Chat System")

# === МОДЕЛИ ДАННЫХ ===

class Message(BaseModel):
    user_id: str
    text: str

class Rating(BaseModel):
    message_id: int
    rating: int  # 1 = хорошо, -1 = плохо

class Correction(BaseModel):
    message_id: int
    corrected_text: str

class UserProfile(BaseModel):
    user_id: str
    name: str = ""
    age: str = ""
    interests: str = ""
    notes: str = ""

class PromptSettings(BaseModel):
    character_name: str
    character_age: str = ""
    character_description: str
    additional_instructions: str = ""
    style_instructions: str = ""
    forbidden_topics: str = ""
    allowed_topics: str = ""

class ImportData(BaseModel):
    examples: list  # [{"user": "...", "assistant": "..."}, ...]

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def build_system_prompt() -> str:
    """Построить системный промпт"""
    return prompt_manager.build_full_prompt()

def build_prompt(user_id: str, new_message: str, rag_context: str = "") -> str:
    """Построить полный промпт для модели"""
    # Системный промпт
    prompt = f"System: {build_system_prompt()}\n\n"
    
    # Информация о собеседнике
    user = db.get_or_create_user(user_id)
    if user.get('name') or user.get('notes'):
        prompt += "Информация о собеседнике:\n"
        if user.get('name'):
            prompt += f"- Его зовут: {user['name']}\n"
        if user.get('age'):
            prompt += f"- Возраст: {user['age']}\n"
        if user.get('interests'):
            prompt += f"- Интересы: {user['interests']}\n"
        if user.get('notes'):
            prompt += f"- Заметки: {user['notes']}\n"
        if user.get('detected_mood'):
            prompt += f"- Текущее настроение: {user['detected_mood']}\n"
        prompt += "\n"
    
    # RAG контекст (примеры похожих ситуаций)
    if rag_context:
        prompt += rag_context
    
    # История переписки
    messages = db.get_recent_messages(user_id, MAX_CONTEXT_MESSAGES)
    if messages:
        prompt += "История переписки:\n"
        for msg in messages:
            prompt += f"Мужчина: {msg['user_message']}\n"
            response = msg.get('corrected_response') or msg['bot_response']
            prompt += f"Ты: {response}\n\n"
    
    # Новое сообщение
    prompt += f"Мужчина: {new_message}\nТы:"
    
    return prompt

def clean_response(response: str) -> str:
    """Очистить ответ от лишнего текста"""
    # Убираем "Ты:" в начале если модель его добавила
    if response.startswith("Ты:"):
        response = response[3:].strip()
    if response.startswith("Ты :"):
        response = response[4:].strip()

    # Убираем если модель начала говорить за мужчину
    stop_patterns = ["Мужчина:", "Привет! Я", "Здравствуй! Я", "Human:", "User:"]

    for pattern in stop_patterns:
        if pattern in response:
            response = response.split(pattern)[0].strip()

    # Убираем повторения (если текст повторяется)
    sentences = response.split('.')
    seen = set()
    unique = []
    for s in sentences:
        s_clean = s.strip().lower()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique.append(s.strip())

    if unique:
        response = '. '.join(unique)
        if not response.endswith(('.', '!', '?', ')', '😊', '😉', '😏')):
            response += '.'

    return response.strip()

def call_ollama(prompt: str) -> str:
    """Вызвать Ollama через API"""
    import json
    import urllib.request

    try:
        data = json.dumps({
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "stop": ["Мужчина:", "\nМужчина:", "Human:", "\nHuman:", "Привет! Я", "\nПривет!"],
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 150
            }
        }).encode('utf-8')

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get("response", "").strip()
            return clean_response(text)

    except Exception as e:
        # Fallback на subprocess
        try:
            result = subprocess.run(
                ["ollama", "run", MODEL_NAME, prompt],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120
            )
            return clean_response(result.stdout.strip())
        except:
            return "Извини, что-то пошло не так..."

# === API ENDPOINTS ===

@app.post("/chat")
async def chat(message: Message):
    """Отправить сообщение и получить ответ"""
    
    # Анализируем сообщение
    analysis = analyzer.analyze_message(message.text)
    detected_name = analysis['detected_name']
    detected_mood = analysis['detected_mood']
    
    # Обновляем профиль если нашли имя
    if detected_name:
        user = db.get_or_create_user(message.user_id)
        if not user.get('name'):
            db.update_user(message.user_id, name=detected_name)
    
    # Обновляем настроение
    db.update_user(message.user_id, detected_mood=detected_mood)
    
    # Получаем RAG контекст
    rag_context = rag.build_rag_context(message.text)
    rag_examples = rag.find_similar_examples(message.text)
    
    # Строим промпт и получаем ответ
    prompt = build_prompt(message.user_id, message.text, rag_context)
    response = call_ollama(prompt)
    
    # Сохраняем в БД
    message_id = db.save_message(
        user_id=message.user_id,
        user_message=message.text,
        bot_response=response,
        detected_name=detected_name,
        detected_mood=detected_mood,
        rag_examples=rag_examples
    )
    
    return {
        "response": response,
        "message_id": message_id,
        "detected_name": detected_name,
        "detected_mood": detected_mood,
        "rag_examples_count": len(rag_examples)
    }

@app.post("/rate")
async def rate_message(rating: Rating):
    """Оценить ответ"""
    msg = db.get_message_by_id(rating.message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Сообщение не найдено")
    
    db.rate_message(rating.message_id, rating.rating)
    
    # Если положительная оценка — добавляем в RAG
    if rating.rating == 1:
        rag.add_example(msg['user_message'], msg['bot_response'])
    
    return {"status": "ok", "message_id": rating.message_id, "rating": rating.rating}

@app.post("/correct")
async def correct_message(correction: Correction):
    """Исправить ответ"""
    msg = db.get_message_by_id(correction.message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Сообщение не найдено")
    
    db.correct_message(correction.message_id, correction.corrected_text)
    
    # Добавляем исправленный вариант в RAG
    rag.add_example(msg['user_message'], correction.corrected_text)
    
    return {"status": "ok", "message_id": correction.message_id}

@app.post("/regenerate")
async def regenerate_response(message: Message):
    """Сгенерировать новый ответ"""
    rag_context = rag.build_rag_context(message.text)
    prompt = build_prompt(message.user_id, message.text, rag_context)
    response = call_ollama(prompt)
    return {"response": response}

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Получить историю сообщений"""
    messages = db.get_user_messages(user_id)
    return {"history": messages}

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Очистить историю"""
    db.clear_user_history(user_id)
    return {"status": "cleared"}

@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Получить профиль пользователя"""
    return db.get_or_create_user(user_id)

@app.post("/profile")
async def update_profile(profile: UserProfile):
    """Обновить профиль"""
    db.update_user(
        profile.user_id,
        name=profile.name,
        age=profile.age,
        interests=profile.interests,
        notes=profile.notes
    )
    return {"status": "updated"}

@app.get("/users")
async def get_users():
    """Список пользователей"""
    return {"users": db.get_all_users()}

@app.get("/stats")
async def get_stats():
    """Статистика системы"""
    db_stats = db.get_stats()
    rag_stats = rag.get_stats()
    return {
        **db_stats,
        "rag_enabled": rag_stats["enabled"],
        "rag_examples": rag_stats["count"],
        "model": MODEL_NAME
    }

# === ОБУЧЕНИЕ ===

@app.post("/export")
async def export_training_data():
    """Экспортировать данные для обучения"""
    result = training.export_for_training()
    return result

@app.get("/export/all")
async def export_all():
    """Экспортировать все сообщения"""
    result = training.export_all_messages()
    return result

@app.get("/export/script")
async def get_training_script():
    """Получить скрипт для обучения"""
    return {"script": training.get_training_script()}

@app.get("/export/download/{filename}")
async def download_export(filename: str):
    """Скачать файл экспорта"""
    filepath = Path(EXPORTS_DIR) / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(filepath, filename=filename)

@app.get("/exports")
async def list_exports():
    """Список экспортов"""
    exports_path = Path(EXPORTS_DIR)
    if not exports_path.exists():
        return {"exports": []}
    
    files = []
    for f in exports_path.glob("*.json"):
        files.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "created": f.stat().st_mtime
        })
    return {"exports": sorted(files, key=lambda x: x['created'], reverse=True)}

# === ПРОМПТ ===

@app.get("/prompt")
async def get_prompt():
    """Получить текущий промпт"""
    return prompt_manager.get_prompt()

@app.post("/prompt")
async def save_prompt(settings: PromptSettings):
    """Сохранить промпт"""
    success = prompt_manager.save_prompt(settings.model_dump())
    if success:
        return {"status": "saved"}
    raise HTTPException(status_code=400, detail="Ошибка сохранения промпта")

@app.post("/prompt/reset")
async def reset_prompt():
    """Сбросить промпт на дефолтный"""
    success = prompt_manager.reset_to_default()
    if success:
        return {"status": "reset"}
    raise HTTPException(status_code=500, detail="Ошибка сброса промпта")

@app.get("/prompt/preview")
async def preview_prompt():
    """Предпросмотр полного промпта"""
    return {"prompt": prompt_manager.build_full_prompt()}

# === ИМПОРТ В RAG ===

@app.post("/import/rag")
async def import_to_rag(data: ImportData):
    """Массовый импорт примеров в RAG"""
    imported = 0
    errors = []

    for i, example in enumerate(data.examples):
        user_msg = example.get("user") or example.get("user_message") or example.get("human")
        assistant_msg = example.get("assistant") or example.get("response") or example.get("bot") or example.get("gpt")

        if user_msg and assistant_msg:
            if rag.add_example(user_msg, assistant_msg):
                imported += 1
            else:
                errors.append(f"Пример {i+1}: ошибка добавления")
        else:
            errors.append(f"Пример {i+1}: отсутствует user или assistant")

    return {
        "success": True,
        "imported": imported,
        "total": len(data.examples),
        "errors": errors[:10]  # Первые 10 ошибок
    }

@app.post("/import/rag/clear")
async def clear_rag():
    """Очистить RAG базу"""
    rag.clear_all()
    return {"status": "cleared", "count": rag.get_stats()["count"]}

@app.get("/rag/examples")
async def get_rag_examples():
    """Получить все примеры из RAG"""
    if rag.collection is None:
        return {"examples": [], "count": 0}

    try:
        # Получаем все записи
        results = rag.collection.get(include=["metadatas"])
        examples = []
        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                examples.append({
                    "user": metadata.get("user_message", ""),
                    "assistant": metadata.get("response", "")
                })
        return {"examples": examples, "count": len(examples)}
    except Exception as e:
        return {"examples": [], "count": 0, "error": str(e)}

# === ВЕБ-ИНТЕРФЕЙС ===

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    return '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>💬 Chat System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #e0e0e0; }
        .container { display: flex; height: 100vh; }
        
        /* Sidebar */
        .sidebar { width: 280px; background: #1a1a2e; display: flex; flex-direction: column; border-right: 1px solid #333; }
        .sidebar-header { padding: 20px; border-bottom: 1px solid #333; }
        .sidebar-header h2 { color: #e94560; margin-bottom: 15px; }
        .new-chat { width: 100%; padding: 12px; background: #e94560; border: none; color: white; 
                    border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: bold; }
        .new-chat:hover { background: #ff6b6b; }
        
        .user-list { flex: 1; overflow-y: auto; padding: 10px; }
        .user-item { padding: 12px; margin: 5px 0; background: #252540; border-radius: 8px; cursor: pointer; }
        .user-item:hover { background: #333355; }
        .user-item.active { background: #e94560; }
        .user-item .name { font-weight: bold; margin-bottom: 4px; }
        .user-item .meta { font-size: 12px; color: #888; }
        
        .sidebar-footer { padding: 15px; border-top: 1px solid #333; }
        .stats-mini { font-size: 12px; color: #888; }
        
        /* Main Chat Area */
        .chat-area { flex: 1; display: flex; flex-direction: column; }
        
        .chat-header { padding: 15px 20px; background: #1a1a2e; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 15px; }
        .chat-header input { padding: 10px; border-radius: 6px; border: 1px solid #444; background: #252540; color: white; width: 200px; }
        .chat-header button { padding: 10px 15px; border-radius: 6px; border: none; cursor: pointer; font-size: 13px; }
        .btn-primary { background: #e94560; color: white; }
        .btn-secondary { background: #444; color: white; }
        .btn-danger { background: #c0392b; color: white; }
        .mood-indicator { padding: 5px 12px; border-radius: 20px; font-size: 12px; background: #333; }
        .mood-позитивное { background: #27ae60; }
        .mood-грустное { background: #3498db; }
        .mood-флирт { background: #e91e63; }
        .mood-агрессия { background: #c0392b; }
        .mood-заинтересованность { background: #f39c12; }
        
        .messages { flex: 1; padding: 20px; overflow-y: auto; }
        .message { margin: 10px 0; max-width: 75%; }
        .message-content { padding: 12px 16px; border-radius: 16px; position: relative; }
        .message.user { margin-left: auto; }
        .message.user .message-content { background: #e94560; border-bottom-right-radius: 4px; }
        .message.bot .message-content { background: #252540; border-bottom-left-radius: 4px; }
        .message-actions { margin-top: 8px; display: flex; gap: 8px; opacity: 0; transition: opacity 0.2s; }
        .message:hover .message-actions { opacity: 1; }
        .message-actions button { padding: 4px 10px; font-size: 12px; border: none; border-radius: 4px; cursor: pointer; background: #333; color: #ccc; }
        .message-actions button:hover { background: #444; }
        .message-actions button.liked { background: #27ae60; color: white; }
        .message-actions button.disliked { background: #c0392b; color: white; }
        .message-meta { font-size: 11px; color: #666; margin-top: 4px; }
        
        .input-area { padding: 20px; background: #1a1a2e; border-top: 1px solid #333; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 14px; border-radius: 8px; border: 1px solid #444; 
                           background: #252540; color: white; font-size: 14px; }
        .input-area button { padding: 14px 28px; background: #e94560; border: none; 
                            color: white; border-radius: 8px; cursor: pointer; font-weight: bold; }
        .input-area button:hover { background: #ff6b6b; }
        
        /* Right Panel */
        .right-panel { width: 320px; background: #1a1a2e; border-left: 1px solid #333; display: flex; flex-direction: column; }
        .panel-tabs { display: flex; border-bottom: 1px solid #333; }
        .panel-tab { flex: 1; padding: 12px; text-align: center; cursor: pointer; background: transparent; border: none; color: #888; font-size: 13px; }
        .panel-tab.active { background: #252540; color: #e94560; border-bottom: 2px solid #e94560; }
        
        .panel-content { flex: 1; overflow-y: auto; padding: 20px; display: none; }
        .panel-content.active { display: block; }
        
        .panel-content h3 { color: #e94560; margin-bottom: 15px; font-size: 14px; }
        .panel-content label { display: block; margin-top: 12px; margin-bottom: 5px; font-size: 13px; color: #888; }
        .panel-content input, .panel-content textarea { 
            width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #444; 
            background: #252540; color: white; font-size: 13px; 
        }
        .panel-content textarea { resize: vertical; min-height: 80px; }
        .panel-content button { margin-top: 15px; width: 100%; }
        
        .stat-card { background: #252540; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
        .stat-card .value { font-size: 24px; font-weight: bold; color: #e94560; }
        .stat-card .label { font-size: 12px; color: #888; margin-top: 4px; }
        
        .export-item { background: #252540; padding: 12px; border-radius: 6px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        .export-item .name { font-size: 13px; }
        .export-item button { padding: 6px 12px; font-size: 12px; }
        
        /* Correction Modal */
        .modal { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); z-index: 1000; align-items: center; justify-content: center; }
        .modal.show { display: flex; }
        .modal-content { background: #1a1a2e; padding: 25px; border-radius: 12px; width: 500px; max-width: 90%; }
        .modal-content h3 { margin-bottom: 15px; color: #e94560; }
        .modal-content textarea { width: 100%; min-height: 120px; margin: 15px 0; }
        .modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
        
        /* Loading */
        .loading { color: #888; font-style: italic; }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1a1a2e; }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="sidebar-header">
                <h2>💬 Чаты</h2>
                <button class="new-chat" onclick="newChat()">+ Новый чат</button>
            </div>
            <div class="user-list" id="userList"></div>
            <div class="sidebar-footer">
                <div class="stats-mini" id="statsMini">Загрузка...</div>
            </div>
        </div>
        
        <!-- Main Chat -->
        <div class="chat-area">
            <div class="chat-header">
                <span>ID:</span>
                <input type="text" id="userId" placeholder="ID пользователя" onchange="loadChat()">
                <button class="btn-primary" onclick="loadChat()">Загрузить</button>
                <button class="btn-secondary" onclick="clearHistory()">🗑️ Очистить</button>
                <div class="mood-indicator" id="moodIndicator">—</div>
            </div>
            <div class="messages" id="messages">
                <div style="text-align: center; color: #666; margin-top: 50px;">
                    Выберите чат или создайте новый
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Введите сообщение..." 
                       onkeypress="if(event.key==='Enter')sendMessage()">
                <button onclick="sendMessage()">Отправить</button>
            </div>
        </div>
        
        <!-- Right Panel -->
        <div class="right-panel">
            <div class="panel-tabs">
                <button class="panel-tab active" onclick="showPanel('profile')">👤</button>
                <button class="panel-tab" onclick="showPanel('prompt')">📝</button>
                <button class="panel-tab" onclick="showPanel('import')">📥</button>
                <button class="panel-tab" onclick="showPanel('stats')">📊</button>
                <button class="panel-tab" onclick="showPanel('training')">🎓</button>
            </div>
            
            <div class="panel-content active" id="panel-profile">
                <h3>Информация о собеседнике</h3>
                <label>Имя:</label>
                <input type="text" id="profileName" placeholder="Определится автоматически">
                <label>Возраст:</label>
                <input type="text" id="profileAge">
                <label>Интересы:</label>
                <input type="text" id="profileInterests">
                <label>Заметки:</label>
                <textarea id="profileNotes" placeholder="Личные заметки о собеседнике..."></textarea>
                <button class="btn-primary" onclick="saveProfile()">💾 Сохранить</button>
            </div>

            <div class="panel-content" id="panel-prompt">
                <h3>📝 Настройки промпта</h3>
                <label>Имя персонажа:</label>
                <input type="text" id="promptName" placeholder="Анна">
                <label>Возраст:</label>
                <input type="text" id="promptAge" placeholder="28">
                <label>Описание персонажа:</label>
                <textarea id="promptDescription" rows="5" placeholder="Ты — женщина по имени {name}..."></textarea>
                <label>Дополнительные инструкции:</label>
                <textarea id="promptAdditional" rows="3" placeholder="Дополнительные правила поведения..."></textarea>
                <label>Стиль общения:</label>
                <textarea id="promptStyle" rows="2" placeholder="Дружелюбный, с юмором..."></textarea>
                <label>Разрешённые темы:</label>
                <textarea id="promptAllowed" rows="2" placeholder="Флирт, романтика, интимные темы..."></textarea>
                <label>Запрещённые темы (опционально):</label>
                <textarea id="promptForbidden" rows="2" placeholder=""></textarea>
                <button class="btn-primary" onclick="savePrompt()">💾 Сохранить промпт</button>
                <button class="btn-secondary" onclick="previewPrompt()" style="margin-top:8px;">👁️ Предпросмотр</button>
                <button class="btn-danger" onclick="resetPrompt()" style="margin-top:8px;">🔄 Сбросить</button>
            </div>

            <div class="panel-content" id="panel-import">
                <h3>📥 Импорт примеров</h3>
                <p style="font-size:12px;color:#888;margin-bottom:10px;">
                    Импортируйте примеры диалогов для мгновенного улучшения ответов (RAG).
                </p>
                <label>JSON с примерами:</label>
                <textarea id="importJson" rows="8" placeholder='[
  {"user": "Привет", "assistant": "Привет! Как дела?"},
  {"user": "Чем занимаешься?", "assistant": "Думаю о тебе 😊"}
]'></textarea>
                <button class="btn-primary" onclick="importExamples()">📥 Импортировать</button>

                <h3 style="margin-top:20px;">Текущие примеры в RAG</h3>
                <div id="ragStats" style="font-size:13px;color:#888;margin-bottom:10px;">Загрузка...</div>
                <div id="ragExamples" style="max-height:200px;overflow-y:auto;"></div>
                <button class="btn-danger" onclick="clearRag()" style="margin-top:10px;">🗑️ Очистить RAG</button>
            </div>

            <div class="panel-content" id="panel-stats">
                <h3>Статистика системы</h3>
                <div class="stat-card">
                    <div class="value" id="statUsers">0</div>
                    <div class="label">Пользователей</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="statMessages">0</div>
                    <div class="label">Сообщений</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="statPositive">0</div>
                    <div class="label">👍 Положительных оценок</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="statCorrections">0</div>
                    <div class="label">✏️ Исправлений</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="statRAG">0</div>
                    <div class="label">📚 Примеров в RAG</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="statReady">0</div>
                    <div class="label">🎓 Готово для обучения</div>
                </div>
                <button class="btn-secondary" onclick="loadStats()" style="margin-top: 10px;">🔄 Обновить</button>
            </div>
            
            <div class="panel-content" id="panel-training">
                <h3>Обучение модели</h3>
                <p style="font-size: 13px; color: #888; margin-bottom: 15px;">
                    Экспортируйте данные с положительными оценками и исправлениями для дообучения модели.
                </p>
                <button class="btn-primary" onclick="exportData()">📦 Экспорт для обучения</button>
                <button class="btn-secondary" onclick="exportAll()" style="margin-top: 10px;">📋 Экспорт всех сообщений</button>
                <button class="btn-secondary" onclick="showScript()" style="margin-top: 10px;">📜 Инструкция RunPod</button>
                
                <h3 style="margin-top: 25px;">Файлы экспорта</h3>
                <div id="exportsList"></div>
            </div>
        </div>
    </div>
    
    <!-- Correction Modal -->
    <div class="modal" id="correctionModal">
        <div class="modal-content">
            <h3>✏️ Исправить ответ</h3>
            <p style="color: #888; font-size: 13px;">Введите правильный вариант ответа:</p>
            <textarea id="correctionText" placeholder="Правильный ответ..."></textarea>
            <div class="modal-actions">
                <button class="btn-secondary" onclick="closeModal()">Отмена</button>
                <button class="btn-primary" onclick="submitCorrection()">Сохранить</button>
            </div>
        </div>
    </div>

    <script>
        let currentUserId = '';
        let currentMessageId = null;
        
        // === ИНИЦИАЛИЗАЦИЯ ===
        loadUsers();
        loadStats();
        loadExports();
        
        // === ПОЛЬЗОВАТЕЛИ ===
        async function loadUsers() {
            const res = await fetch('/users');
            const data = await res.json();
            const list = document.getElementById('userList');
            list.innerHTML = data.users.map(u => `
                <div class="user-item ${u.user_id === currentUserId ? 'active' : ''}" 
                     onclick="selectUser('${u.user_id}')">
                    <div class="name">${u.name || u.user_id}</div>
                    <div class="meta">${u.message_count || 0} сообщений</div>
                </div>
            `).join('') || '<div style="color:#666;text-align:center;padding:20px;">Нет чатов</div>';
        }
        
        function selectUser(userId) {
            document.getElementById('userId').value = userId;
            loadChat();
        }
        
        function newChat() {
            const id = prompt('Введите ID нового пользователя:');
            if (id && id.trim()) {
                document.getElementById('userId').value = id.trim();
                loadChat();
            }
        }
        
        // === ЧАТ ===
        async function loadChat() {
            currentUserId = document.getElementById('userId').value.trim();
            if (!currentUserId) return;
            
            // История
            const histRes = await fetch(`/history/${currentUserId}`);
            const histData = await histRes.json();
            
            const messagesDiv = document.getElementById('messages');
            if (histData.history.length === 0) {
                messagesDiv.innerHTML = '<div style="text-align:center;color:#666;margin-top:50px;">Начните диалог</div>';
            } else {
                messagesDiv.innerHTML = histData.history.map(m => `
                    <div class="message user">
                        <div class="message-content">${escapeHtml(m.user_message)}</div>
                        ${m.detected_name ? `<div class="message-meta">👤 Определено имя: ${m.detected_name}</div>` : ''}
                    </div>
                    <div class="message bot">
                        <div class="message-content">${escapeHtml(m.corrected_response || m.bot_response)}</div>
                        <div class="message-actions">
                            <button class="${m.rating === 1 ? 'liked' : ''}" onclick="rate(${m.id}, 1)">👍</button>
                            <button class="${m.rating === -1 ? 'disliked' : ''}" onclick="rate(${m.id}, -1)">👎</button>
                            <button onclick="openCorrection(${m.id}, '${escapeJs(m.corrected_response || m.bot_response)}')">✏️</button>
                            <button onclick="regenerate('${escapeJs(m.user_message)}')">🔄</button>
                        </div>
                        ${m.corrected_response ? '<div class="message-meta">✏️ Исправлено</div>' : ''}
                    </div>
                `).join('');
            }
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Профиль
            const profRes = await fetch(`/profile/${currentUserId}`);
            const prof = await profRes.json();
            document.getElementById('profileName').value = prof.name || '';
            document.getElementById('profileAge').value = prof.age || '';
            document.getElementById('profileInterests').value = prof.interests || '';
            document.getElementById('profileNotes').value = prof.notes || '';
            
            // Настроение
            const moodEl = document.getElementById('moodIndicator');
            const mood = prof.detected_mood || 'нейтральное';
            moodEl.textContent = mood;
            moodEl.className = 'mood-indicator mood-' + mood;
            
            loadUsers();
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message || !currentUserId) return;
            
            input.value = '';
            
            const messagesDiv = document.getElementById('messages');
            messagesDiv.innerHTML += `
                <div class="message user">
                    <div class="message-content">${escapeHtml(message)}</div>
                </div>
                <div class="message bot">
                    <div class="message-content loading">Печатает...</div>
                </div>
            `;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({user_id: currentUserId, text: message})
            });
            const data = await res.json();
            
            // Обновляем чат
            loadChat();
            loadStats();
        }
        
        async function rate(messageId, rating) {
            await fetch('/rate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message_id: messageId, rating: rating})
            });
            loadChat();
            loadStats();
        }
        
        function openCorrection(messageId, currentText) {
            currentMessageId = messageId;
            document.getElementById('correctionText').value = currentText;
            document.getElementById('correctionModal').classList.add('show');
        }
        
        function closeModal() {
            document.getElementById('correctionModal').classList.remove('show');
            currentMessageId = null;
        }
        
        async function submitCorrection() {
            const text = document.getElementById('correctionText').value.trim();
            if (!text || !currentMessageId) return;
            
            await fetch('/correct', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message_id: currentMessageId, corrected_text: text})
            });
            
            closeModal();
            loadChat();
            loadStats();
        }
        
        async function regenerate(userMessage) {
            const messagesDiv = document.getElementById('messages');
            messagesDiv.innerHTML += `
                <div class="message bot">
                    <div class="message-content loading">Генерирую новый ответ...</div>
                </div>
            `;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            const res = await fetch('/regenerate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({user_id: currentUserId, text: userMessage})
            });
            const data = await res.json();
            
            alert('Новый вариант:\\n\\n' + data.response);
            loadChat();
        }
        
        async function clearHistory() {
            if (!currentUserId || !confirm('Очистить историю?')) return;
            await fetch(`/history/${currentUserId}`, {method: 'DELETE'});
            loadChat();
        }
        
        // === ПРОФИЛЬ ===
        async function saveProfile() {
            if (!currentUserId) return alert('Выберите пользователя');
            
            await fetch('/profile', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    user_id: currentUserId,
                    name: document.getElementById('profileName').value,
                    age: document.getElementById('profileAge').value,
                    interests: document.getElementById('profileInterests').value,
                    notes: document.getElementById('profileNotes').value
                })
            });
            alert('Профиль сохранён!');
            loadUsers();
        }
        
        // === СТАТИСТИКА ===
        async function loadStats() {
            const res = await fetch('/stats');
            const s = await res.json();
            
            document.getElementById('statUsers').textContent = s.total_users;
            document.getElementById('statMessages').textContent = s.total_messages;
            document.getElementById('statPositive').textContent = s.positive_ratings;
            document.getElementById('statCorrections').textContent = s.corrections;
            document.getElementById('statRAG').textContent = s.rag_examples;
            document.getElementById('statReady').textContent = s.training_ready;
            
            document.getElementById('statsMini').textContent = 
                `${s.total_users} чатов · ${s.total_messages} сообщений · ${s.training_ready} для обучения`;
        }
        
        // === ОБУЧЕНИЕ ===
        async function exportData() {
            const res = await fetch('/export', {method: 'POST'});
            const data = await res.json();
            
            if (data.success) {
                alert(`✅ Экспортировано ${data.count} сообщений в ${data.filename}`);
                loadExports();
            } else {
                alert('❌ ' + data.error);
            }
        }
        
        async function exportAll() {
            const res = await fetch('/export/all');
            const data = await res.json();
            
            if (data.success) {
                alert(`✅ Экспортировано ${data.count} сообщений в ${data.filename}`);
                loadExports();
            } else {
                alert('❌ ' + data.error);
            }
        }
        
        async function loadExports() {
            const res = await fetch('/exports');
            const data = await res.json();
            
            const list = document.getElementById('exportsList');
            list.innerHTML = data.exports.map(e => `
                <div class="export-item">
                    <span class="name">${e.filename}</span>
                    <a href="/export/download/${e.filename}" download>
                        <button class="btn-secondary">⬇️</button>
                    </a>
                </div>
            `).join('') || '<div style="color:#666;font-size:13px;">Нет экспортов</div>';
        }
        
        async function showScript() {
            const res = await fetch('/export/script');
            const data = await res.json();
            
            const w = window.open('', '_blank');
            w.document.write(`<pre style="background:#1a1a2e;color:#e0e0e0;padding:20px;font-family:monospace;white-space:pre-wrap;">${data.script}</pre>`);
        }
        
        // === ПАНЕЛИ ===
        function showPanel(name) {
            document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel-content').forEach(p => p.classList.remove('active'));

            document.querySelector(`.panel-tab[onclick="showPanel('${name}')"]`).classList.add('active');
            document.getElementById('panel-' + name).classList.add('active');

            // Загружаем данные для вкладки
            if (name === 'prompt') loadPrompt();
            if (name === 'import') loadRagExamples();
        }

        // === ПРОМПТ ===
        async function loadPrompt() {
            const res = await fetch('/prompt');
            const p = await res.json();
            document.getElementById('promptName').value = p.character_name || '';
            document.getElementById('promptAge').value = p.character_age || '';
            document.getElementById('promptDescription').value = p.character_description || '';
            document.getElementById('promptAdditional').value = p.additional_instructions || '';
            document.getElementById('promptStyle').value = p.style_instructions || '';
            document.getElementById('promptAllowed').value = p.allowed_topics || '';
            document.getElementById('promptForbidden').value = p.forbidden_topics || '';
        }

        async function savePrompt() {
            const data = {
                character_name: document.getElementById('promptName').value,
                character_age: document.getElementById('promptAge').value,
                character_description: document.getElementById('promptDescription').value,
                additional_instructions: document.getElementById('promptAdditional').value,
                style_instructions: document.getElementById('promptStyle').value,
                allowed_topics: document.getElementById('promptAllowed').value,
                forbidden_topics: document.getElementById('promptForbidden').value
            };

            if (!data.character_name || !data.character_description) {
                return alert('Заполните имя и описание персонажа!');
            }

            const res = await fetch('/prompt', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });

            if (res.ok) {
                alert('✅ Промпт сохранён!');
            } else {
                alert('❌ Ошибка сохранения');
            }
        }

        async function previewPrompt() {
            const res = await fetch('/prompt/preview');
            const data = await res.json();
            const w = window.open('', '_blank', 'width=600,height=500');
            w.document.write(`
                <html><head><title>Предпросмотр промпта</title></head>
                <body style="background:#1a1a2e;color:#e0e0e0;padding:20px;font-family:monospace;">
                <h2>Текущий системный промпт:</h2>
                <pre style="white-space:pre-wrap;background:#252540;padding:15px;border-radius:8px;">${data.prompt}</pre>
                </body></html>
            `);
        }

        async function resetPrompt() {
            if (!confirm('Сбросить промпт на значения по умолчанию?')) return;
            await fetch('/prompt/reset', {method: 'POST'});
            loadPrompt();
            alert('✅ Промпт сброшен');
        }

        // === ИМПОРТ RAG ===
        async function loadRagExamples() {
            const res = await fetch('/rag/examples');
            const data = await res.json();

            document.getElementById('ragStats').textContent = `Примеров в базе: ${data.count}`;

            const list = document.getElementById('ragExamples');
            if (data.examples.length === 0) {
                list.innerHTML = '<div style="color:#666;font-size:12px;">Нет примеров</div>';
            } else {
                list.innerHTML = data.examples.slice(0, 20).map(e => `
                    <div style="background:#252540;padding:8px;margin:5px 0;border-radius:4px;font-size:12px;">
                        <div style="color:#888;">👤 ${escapeHtml(e.user.substring(0,50))}...</div>
                        <div style="color:#e94560;">🤖 ${escapeHtml(e.assistant.substring(0,50))}...</div>
                    </div>
                `).join('');
            }
        }

        async function importExamples() {
            const jsonText = document.getElementById('importJson').value.trim();
            if (!jsonText) return alert('Вставьте JSON с примерами');

            let examples;
            try {
                examples = JSON.parse(jsonText);
                if (!Array.isArray(examples)) {
                    examples = [examples];
                }
            } catch (e) {
                return alert('❌ Некорректный JSON: ' + e.message);
            }

            const res = await fetch('/import/rag', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({examples: examples})
            });
            const data = await res.json();

            if (data.success) {
                alert(`✅ Импортировано: ${data.imported} из ${data.total}`);
                document.getElementById('importJson').value = '';
                loadRagExamples();
                loadStats();
            } else {
                alert('❌ Ошибка импорта');
            }
        }

        async function clearRag() {
            if (!confirm('Очистить ВСЮ базу примеров RAG?')) return;
            await fetch('/import/rag/clear', {method: 'POST'});
            loadRagExamples();
            loadStats();
            alert('✅ RAG очищен');
        }

        // === УТИЛИТЫ ===
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function escapeJs(text) {
            return text.replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\\n/g, '\\\\n');
        }
    </script>
</body>
</html>'''

if __name__ == "__main__":
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║              💬 CHAT SYSTEM ЗАПУЩЕН                       ║
╠═══════════════════════════════════════════════════════════╣
║  Откройте в браузере: http://{HOST}:{PORT}                  ║
║  Модель: {MODEL_NAME}                                          ║
║  Для остановки нажмите Ctrl+C                             ║
╚═══════════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host=HOST, port=PORT)
