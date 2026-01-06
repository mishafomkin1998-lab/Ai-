import json
from datetime import datetime
from pathlib import Path
from database import get_training_data, save_export_record
from config import EXPORTS_DIR, CHARACTER_DESCRIPTION, CHARACTER_NAME, CHARACTER_AGE

# Создаём папку
Path(EXPORTS_DIR).mkdir(parents=True, exist_ok=True)

# Системный промпт для обучения
SYSTEM_PROMPT = CHARACTER_DESCRIPTION.format(name=CHARACTER_NAME, age=CHARACTER_AGE)

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
    
    # Формируем в формате ShareGPT с системным промптом
    sharegpt_data = []
    for user_id, messages in user_dialogs.items():
        # Начинаем с системного промпта
        conversations = [{"from": "system", "value": SYSTEM_PROMPT}]

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
# Модель: Mistral-Nemo 12B (без цензуры, хороший русский)
# Оптимизированные параметры для качественного обучения

# 1. Создайте под на RunPod:
#    - GPU: A100 (80GB) или A40 (48GB)
#    - Container Disk: 50 GB
#    - Volume Disk: 100 GB
#    - Template: RunPod Pytorch 2.1

# 2. Загрузите файл training_data_XXX.json в /workspace/training_data.json

# 3. Выполните эту команду:

cd /workspace && pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && cat > train.py << 'EOF'
import os
os.environ['WANDB_DISABLED'] = 'true'

from unsloth import FastLanguageModel
import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# === КОНФИГУРАЦИЯ ===
MODEL_NAME = "unsloth/Mistral-Nemo-Instruct-2407"  # Без цензуры, хороший русский
MAX_SEQ_LENGTH = 4096  # Mistral-Nemo поддерживает до 128K, но 4K достаточно
LOAD_IN_4BIT = True

# LoRA параметры (оптимизированные)
LORA_R = 64              # Ранг адаптера
LORA_ALPHA = 128         # Scaling factor (обычно 2*r)
LORA_DROPOUT = 0.05      # Регуляризация против переобучения

# Параметры обучения (консервативные для качества)
EPOCHS = 2               # Меньше эпох = меньше переобучения
LEARNING_RATE = 1e-4     # Консервативный LR
BATCH_SIZE = 2
GRAD_ACCUM = 4           # Эффективный batch = 2*4 = 8
WARMUP_RATIO = 0.1       # 10% warmup

print("=" * 50)
print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ AI-КОМПАНЬОНА")
print("=" * 50)

print("\\n1. Загружаем Mistral-Nemo 12B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=None,  # Auto-detect
)

print("\\n2. Настраиваем LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print("\\n3. Загружаем данные...")
with open("training_data.json", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# Форматирование для Mistral-Nemo (ChatML формат)
def format_conversation(example):
    text = ""
    for msg in example["conversations"]:
        role = msg["from"]
        content = msg["value"]

        if role == "system":
            text += f"<|im_start|>system\\n{content}<|im_end|>\\n"
        elif role == "human":
            text += f"<|im_start|>user\\n{content}<|im_end|>\\n"
        elif role == "gpt":
            text += f"<|im_start|>assistant\\n{content}<|im_end|>\\n"

    return {"text": text}

dataset = dataset.map(format_conversation)
print(f"   Загружено {len(dataset)} диалогов")

# Показываем пример
print("\\n   Пример форматирования:")
print("-" * 40)
print(dataset[0]["text"][:500] + "...")
print("-" * 40)

print("\\n4. Начинаем обучение...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=42,
    ),
)

trainer.train()

print("\\n5. Сохраняем в GGUF формате...")
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m"  # Хороший баланс качества и размера
)

print("\\n" + "=" * 50)
print("ГОТОВО!")
print("=" * 50)
print("\\nФайлы сохранены в /workspace/model_gguf/")
print("Скачайте файл *.gguf и используйте его в Ollama")
EOF

python train.py

# 4. Скачайте файл /workspace/model_gguf/*.gguf

# 5. Создайте Modelfile на своём компьютере:
cat > Modelfile << 'MODELFILE'
FROM ./model_gguf.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
MODELFILE

# 6. Импортируйте в Ollama:
#    ollama rm mymodel 2>/dev/null
#    ollama create mymodel -f Modelfile
#    ollama run mymodel "Привет!"
'''
