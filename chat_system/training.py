import json
from datetime import datetime
from pathlib import Path
from database import get_training_data, save_export_record
from config import EXPORTS_DIR

# Создаём папку
Path(EXPORTS_DIR).mkdir(parents=True, exist_ok=True)

def export_for_training() -> dict:
    """Экспортировать данные для обучения в формате ShareGPT"""
    
    # Получаем данные (положительные оценки + исправления)
    data = get_training_data()
    
    if not data:
        return {
            "success": False,
            "error": "Нет данных для экспорта. Оцените ответы 👍 или исправьте их.",
            "count": 0
        }
    
    # Группируем по пользователям для создания диалогов
    user_dialogs = {}
    for item in data:
        user_id = item['user_id']
        if user_id not in user_dialogs:
            user_dialogs[user_id] = []
        user_dialogs[user_id].append({
            "user": item['user_message'],
            "assistant": item['response']
        })
    
    # Формируем в формате ShareGPT
    sharegpt_data = []
    for user_id, messages in user_dialogs.items():
        conversations = []
        for msg in messages:
            conversations.append({"from": "human", "value": msg["user"]})
            conversations.append({"from": "gpt", "value": msg["assistant"]})
        
        sharegpt_data.append({"conversations": conversations})
    
    # Сохраняем файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data_{timestamp}.json"
    filepath = Path(EXPORTS_DIR) / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    
    # Записываем в БД
    save_export_record(filename, len(data))
    
    return {
        "success": True,
        "filename": filename,
        "filepath": str(filepath),
        "count": len(data),
        "dialogs": len(sharegpt_data)
    }

def export_all_messages() -> dict:
    """Экспортировать ВСЕ сообщения (для анализа)"""
    from database import get_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_id, user_message, bot_response, rating, corrected_response, 
               detected_name, detected_mood, created_at
        FROM messages
        ORDER BY created_at ASC
    ''')
    
    data = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if not data:
        return {"success": False, "error": "Нет сообщений", "count": 0}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_messages_{timestamp}.json"
    filepath = Path(EXPORTS_DIR) / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return {
        "success": True,
        "filename": filename,
        "filepath": str(filepath),
        "count": len(data)
    }

def get_training_script() -> str:
    """Получить скрипт для обучения на RunPod"""
    return '''# === ИНСТРУКЦИЯ ПО ОБУЧЕНИЮ НА RUNPOD ===

# 1. Создайте под на RunPod:
#    - GPU: A100 или A40
#    - Container Disk: 50 GB
#    - Volume Disk: 100 GB

# 2. Загрузите файл training_data_XXX.json в /workspace/

# 3. Выполните эту команду:

cd /workspace && pip install unsloth && cat > train.py << 'EOF'
import os
os.environ['WANDB_DISABLED'] = 'true'

from unsloth import FastLanguageModel
import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

print("1. Загружаем модель...")
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=64, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_alpha=128, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth")

print("2. Загружаем данные...")
with open("training_data.json") as f: data = json.load(f)
dataset = Dataset.from_list(data)
def fmt(ex):
    t = ""
    for m in ex["conversations"]:
        if m["from"]=="human": t += f"<|user|>\\n{m['value']}</s>\\n"
        elif m["from"]=="gpt": t += f"<|assistant|>\\n{m['value']}</s>\\n"
    return {"text": t}
dataset = dataset.map(fmt)
print(f"   Загружено {len(dataset)} диалогов")

print("3. Обучаем...")
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text", max_seq_length=2048, args=TrainingArguments(output_dir="./output", per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=2e-4, bf16=True, logging_steps=10, save_strategy="epoch", optim="adamw_8bit"))
trainer.train()

print("4. Сохраняем GGUF...")
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")
print("=== ГОТОВО! ===")
EOF
python train.py

# 4. Скачайте файл из /workspace/*.gguf

# 5. Замените модель в Ollama:
#    ollama rm mymodel
#    ollama create mymodel -f Modelfile
'''
