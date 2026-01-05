try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB не установлен - RAG отключён")

from pathlib import Path
import hashlib
from config import CHROMA_DIR, RAG_TOP_K, RAG_MIN_SIMILARITY, RAG_ENABLED

# Создаём папку
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

# Инициализация ChromaDB
client = None
collection = None

if CHROMADB_AVAILABLE and RAG_ENABLED:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(
            name="dialog_examples",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ RAG инициализирован. Примеров в базе: {collection.count()}")
    except Exception as e:
        print(f"⚠️ Ошибка инициализации RAG: {e}")
        collection = None
else:
    print("⚠️ RAG отключён (chromadb не доступен)")

def generate_id(text: str) -> str:
    """Генерировать уникальный ID для текста"""
    return hashlib.md5(text.encode()).hexdigest()[:16]

def add_example(user_message: str, good_response: str, metadata: dict = None):
    """Добавить пример хорошего ответа в базу"""
    if not RAG_ENABLED or collection is None:
        return False
    
    try:
        example_id = generate_id(user_message + good_response)
        
        # Проверяем нет ли уже такого
        existing = collection.get(ids=[example_id])
        if existing and existing['ids']:
            # Обновляем
            collection.update(
                ids=[example_id],
                documents=[user_message],
                metadatas=[{
                    "response": good_response,
                    "user_message": user_message,
                    **(metadata or {})
                }]
            )
        else:
            # Добавляем новый
            collection.add(
                ids=[example_id],
                documents=[user_message],
                metadatas=[{
                    "response": good_response,
                    "user_message": user_message,
                    **(metadata or {})
                }]
            )
        return True
    except Exception as e:
        print(f"Ошибка добавления примера в RAG: {e}")
        return False

def find_similar_examples(user_message: str, top_k: int = None) -> list:
    """Найти похожие примеры для сообщения"""
    if not RAG_ENABLED or collection is None or collection.count() == 0:
        return []
    
    if top_k is None:
        top_k = RAG_TOP_K
    
    try:
        results = collection.query(
            query_texts=[user_message],
            n_results=min(top_k, collection.count())
        )
        
        examples = []
        if results and results['metadatas'] and results['distances']:
            for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                # Конвертируем distance в similarity (для cosine: similarity = 1 - distance)
                similarity = 1 - distance
                
                if similarity >= RAG_MIN_SIMILARITY:
                    examples.append({
                        'user_message': metadata.get('user_message', ''),
                        'response': metadata.get('response', ''),
                        'similarity': round(similarity, 3)
                    })
        
        return examples
    except Exception as e:
        print(f"Ошибка поиска в RAG: {e}")
        return []

def remove_example(user_message: str, response: str):
    """Удалить пример из базы"""
    if collection is None:
        return False
    
    try:
        example_id = generate_id(user_message + response)
        collection.delete(ids=[example_id])
        return True
    except Exception as e:
        print(f"Ошибка удаления из RAG: {e}")
        return False

def get_stats() -> dict:
    """Статистика RAG"""
    if collection is None:
        return {"enabled": False, "count": 0}
    
    return {
        "enabled": RAG_ENABLED,
        "count": collection.count()
    }

def import_from_training_data(data: list):
    """Импортировать данные из тренировочного набора"""
    if not RAG_ENABLED or collection is None:
        return 0
    
    imported = 0
    for item in data:
        if isinstance(item, dict):
            user_msg = item.get('user_message') or item.get('user')
            response = item.get('response') or item.get('assistant')
            
            if user_msg and response:
                if add_example(user_msg, response):
                    imported += 1
    
    return imported

def clear_all():
    """Очистить всю базу RAG"""
    global collection
    if collection is None or client is None:
        return

    try:
        client.delete_collection("dialog_examples")
        collection = client.create_collection(
            name="dialog_examples",
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ RAG база очищена")
    except Exception as e:
        print(f"Ошибка очистки RAG: {e}")

def build_rag_context(user_message: str) -> str:
    """Построить контекст из RAG для промпта"""
    examples = find_similar_examples(user_message)
    
    if not examples:
        return ""
    
    context = "\n--- Примеры похожих ситуаций ---\n"
    for ex in examples:
        context += f"Мужчина: {ex['user_message']}\n"
        context += f"Хороший ответ: {ex['response']}\n\n"
    context += "--- Конец примеров ---\n\n"
    
    return context
