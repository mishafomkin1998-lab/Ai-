"""
Клиент для работы с OpenRouter API
"""

import requests
import os
from pathlib import Path


def load_api_key() -> str | None:
    """Загрузить API ключ из .env или переменной окружения"""
    # Сначала из переменной окружения
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    # Затем из .env файла
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip('"\'')

    return None


class OpenRouterClient:
    """Клиент для OpenRouter API"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or load_api_key()
        self.model = model or "gryphe/mythomax-l2-13b"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            raise ValueError(
                "API ключ не найден! Создай файл .env с OPENROUTER_API_KEY=..."
            )

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        max_tokens: int = 200,
    ) -> str:
        """
        Отправить сообщения и получить ответ

        Args:
            messages: Список сообщений [{"role": "system/user/assistant", "content": "..."}]
            temperature: Креативность (0.0 - 1.0)
            max_tokens: Максимум токенов в ответе

        Returns:
            Текст ответа от модели
        """
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # Убираем артефакты типа <s>
                content = content.replace("<s>", "").replace("</s>", "").strip()
                return content
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("error", {}).get("message", response.text)
                return f"Ошибка API: {error_msg}"

        except requests.Timeout:
            return "Извини, задумалась... Повтори пожалуйста?"
        except Exception as e:
            return f"Ошибка: {str(e)}"

    def simple_chat(self, system_prompt: str, user_message: str) -> str:
        """Простой вызов с системным промптом и сообщением пользователя"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.chat(messages)

    def chat_with_history(
        self,
        system_prompt: str,
        history: list[tuple[str, str]],
        user_message: str,
    ) -> str:
        """
        Вызов с историей переписки

        Args:
            system_prompt: Системный промпт
            history: История [(user_msg, bot_msg), ...]
            user_message: Новое сообщение пользователя
        """
        messages = [{"role": "system", "content": system_prompt}]

        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        messages.append({"role": "user", "content": user_message})

        return self.chat(messages)


# Глобальный клиент (инициализируется при первом использовании)
_client: OpenRouterClient | None = None


def get_client(model: str = None) -> OpenRouterClient:
    """Получить клиент API"""
    global _client
    if _client is None:
        _client = OpenRouterClient(model=model)
    return _client


def call_api(system_prompt: str, user_message: str, history: list = None) -> str:
    """
    Главная функция для вызова API

    Args:
        system_prompt: Системный промпт (описание персонажа)
        user_message: Сообщение от пользователя
        history: История переписки [(user, bot), ...]

    Returns:
        Ответ от модели
    """
    client = get_client()

    if history:
        return client.chat_with_history(system_prompt, history, user_message)
    else:
        return client.simple_chat(system_prompt, user_message)
