import asyncio
from typing import Any

from telegram import Update
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.ext import ContextTypes

from core.dialogue_manager import DialogueManager

# Простое in-memory хранилище контекста пользователей
USER_CONTEXTS: dict[int, dict[str, Any]] = {}

# Глобальный экземпляр DialogueManager
dialogue_manager: DialogueManager | None = None


def set_dialogue_manager(manager: DialogueManager) -> None:
    global dialogue_manager
    dialogue_manager = manager


async def safe_reply_text(
    update: Update,
    text: str,
    max_attempts: int = 3,
) -> None:
    """
    Безопасная отправка сообщения с повторными попытками при сетевых таймаутах.
    """
    if update.message is None:
        return

    for attempt in range(1, max_attempts + 1):
        try:
            await update.message.reply_text(text)
            return

        except RetryAfter as exc:
            wait_seconds = int(getattr(exc, "retry_after", 3))
            print(f"RetryAfter: waiting {wait_seconds}s before retry.")
            await asyncio.sleep(wait_seconds)

        except TimedOut:
            print(f"TimedOut while sending message. Attempt {attempt}/{max_attempts}")
            if attempt == max_attempts:
                raise
            await asyncio.sleep(2)

        except NetworkError as exc:
            print(f"NetworkError while sending message: {exc}. Attempt {attempt}/{max_attempts}")
            if attempt == max_attempts:
                raise
            await asyncio.sleep(2)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await safe_reply_text(
        update,
        "Привет! Я чат-бот, который умеет поддерживать диалог и помогать с выбором кофемашин.\n"
        "Можешь просто написать мне сообщение."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await safe_reply_text(
        update,
        "Я умею:\n"
        "- поддерживать обычный диалог\n"
        "- обсуждать кофе\n"
        "- подбирать кофемашины\n"
        "- сравнивать модели\n"
        "- предлагать варианты по бюджету"
    )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_user is None:
        return

    if dialogue_manager is None:
        await safe_reply_text(update, "Ошибка: dialogue manager не инициализирован.")
        return

    user_id = update.effective_user.id
    user_text = update.message.text or ""

    user_context = USER_CONTEXTS.get(user_id, {})

    result = dialogue_manager.process_message(user_text, user_context)

    USER_CONTEXTS[user_id] = result["context"]

    await safe_reply_text(update, result["reply"])