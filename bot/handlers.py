import asyncio
from typing import Any

from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.ext import ContextTypes
from telegram import Update

from core.dialogue_manager import DialogueManager

from uuid import uuid4

from config import (
    ENABLE_VOICE_REPLY,
    TELEGRAM_REPLY_MAX_ATTEMPTS,
    TELEGRAM_RETRY_AFTER_FALLBACK_SECONDS,
    TELEGRAM_SEND_RETRY_DELAY_SECONDS,
    TMP_DIR,
)
from core.voice_service import VoiceService

# Простое in-memory хранилище контекста пользователей
USER_CONTEXTS: dict[int, dict[str, Any]] = {}

# Глобальный экземпляр DialogueManager
dialogue_manager: DialogueManager | None = None
voice_service: VoiceService | None = None


def set_dialogue_manager(manager: DialogueManager) -> None:
    global dialogue_manager
    dialogue_manager = manager

def set_voice_service(service: VoiceService) -> None:
    global voice_service
    voice_service = service


async def safe_reply_text(
    update: Update,
    text: str,
    max_attempts: int = TELEGRAM_REPLY_MAX_ATTEMPTS,
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
            wait_seconds = int(getattr(exc, "retry_after", TELEGRAM_RETRY_AFTER_FALLBACK_SECONDS))
            print(f"RetryAfter: waiting {wait_seconds}s before retry.")
            await asyncio.sleep(wait_seconds)

        except TimedOut:
            print(f"TimedOut while sending message. Attempt {attempt}/{max_attempts}")
            if attempt == max_attempts:
                raise
            await asyncio.sleep(TELEGRAM_SEND_RETRY_DELAY_SECONDS)

        except NetworkError as exc:
            print(f"NetworkError while sending message: {exc}. Attempt {attempt}/{max_attempts}")
            if attempt == max_attempts:
                raise
            await asyncio.sleep(TELEGRAM_SEND_RETRY_DELAY_SECONDS)


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

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_user is None:
        return

    if dialogue_manager is None:
        await safe_reply_text(update, "Ошибка: dialogue manager не инициализирован.")
        return

    if voice_service is None:
        await safe_reply_text(update, "Ошибка: voice service не инициализирован.")
        return

    if update.message.voice is None:
        await safe_reply_text(update, "Не удалось получить голосовое сообщение.")
        return

    user_id = update.effective_user.id
    user_context = USER_CONTEXTS.get(user_id, {})

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    voice_id = uuid4().hex
    input_path = TMP_DIR / f"{voice_id}_input.ogg"
    output_path = TMP_DIR / f"{voice_id}_reply.mp3"

    try:
        tg_file = await update.message.voice.get_file()
        await tg_file.download_to_drive(custom_path=str(input_path))

        recognized_text = voice_service.transcribe(input_path)

        if not recognized_text:
            await safe_reply_text(update, "Не удалось распознать речь. Попробуй сказать чуть чётче.")
            return

        result = dialogue_manager.process_message(recognized_text, user_context)
        USER_CONTEXTS[user_id] = result["context"]

        await safe_reply_text(
            update,
            f"Ты сказал: {recognized_text}\n\n{result['reply']}"
        )

        if ENABLE_VOICE_REPLY:
            mp3_path = await voice_service.synthesize_to_mp3(result["reply"], output_path)
            if mp3_path and mp3_path.exists():
                with mp3_path.open("rb") as audio_file:
                    await update.message.reply_audio(audio=audio_file)

    except Exception as exc:
        print(f"Voice handler error: {exc}")
        await safe_reply_text(update, "Не получилось обработать голосовое сообщение.")

    finally:
        for path in (input_path, output_path):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
