from telegram import Update
from telegram.error import TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from bot.handlers import (
    handle_text_message,
    help_command,
    handle_voice_message,
    set_dialogue_manager,
    set_voice_service,
    start_command,
)
from config import (
    TELEGRAM_CONNECT_TIMEOUT,
    TELEGRAM_CONNECTION_POOL_SIZE,
    TELEGRAM_POOL_TIMEOUT,
    TELEGRAM_READ_TIMEOUT,
    TELEGRAM_UPDATES_CONNECTION_POOL_SIZE,
    TELEGRAM_WRITE_TIMEOUT,
)
from core.voice_service import VoiceService
from core.dialogue_manager import DialogueManager


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Глобальный обработчик ошибок.
    Не даёт приложению молча "подвисать" после исключений.
    """
    print("=== Telegram error handler ===")
    print(f"Update: {update}")
    print(f"Error: {context.error}")

    # Если ошибка произошла на сообщении, пробуем вежливо уведомить пользователя
    if isinstance(update, Update) and update.message is not None:
        try:
            if isinstance(context.error, TimedOut):
                await update.message.reply_text(
                    "Сеть Telegram сейчас отвечает слишком долго. Попробуй отправить сообщение ещё раз."
                )
            else:
                await update.message.reply_text(
                    "Произошла временная ошибка. Попробуй ещё раз через пару секунд."
                )
        except Exception as nested_error:
            print(f"Failed to notify user about error: {nested_error}")


def create_application(
        bot_token: str,
        dialogue_manager: DialogueManager,
        voice_service: VoiceService,
) -> Application:
    set_dialogue_manager(dialogue_manager)
    set_voice_service(voice_service)

    application = (
        Application.builder()
        .token(bot_token)
        # Таймауты для обычных запросов бота
        .connect_timeout(TELEGRAM_CONNECT_TIMEOUT)
        .read_timeout(TELEGRAM_READ_TIMEOUT)
        .write_timeout(TELEGRAM_WRITE_TIMEOUT)
        .pool_timeout(TELEGRAM_POOL_TIMEOUT)
        .connection_pool_size(TELEGRAM_CONNECTION_POOL_SIZE)
        # Таймауты для long polling (getUpdates)
        .get_updates_connect_timeout(TELEGRAM_CONNECT_TIMEOUT)
        .get_updates_read_timeout(TELEGRAM_READ_TIMEOUT)
        .get_updates_write_timeout(TELEGRAM_WRITE_TIMEOUT)
        .get_updates_pool_timeout(TELEGRAM_POOL_TIMEOUT)
        .get_updates_connection_pool_size(TELEGRAM_UPDATES_CONNECTION_POOL_SIZE)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    application.add_error_handler(error_handler)

    return application
