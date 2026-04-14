from telegram import Update
from telegram.error import TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from bot.handlers import (
    handle_text_message,
    help_command,
    set_dialogue_manager,
    start_command,
)
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


def create_application(bot_token: str, dialogue_manager: DialogueManager) -> Application:
    set_dialogue_manager(dialogue_manager)

    application = (
        Application.builder()
        .token(bot_token)
        # Таймауты для обычных запросов бота
        .connect_timeout(15.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .connection_pool_size(8)
        # Таймауты для long polling (getUpdates)
        .get_updates_connect_timeout(15.0)
        .get_updates_read_timeout(30.0)
        .get_updates_write_timeout(30.0)
        .get_updates_pool_timeout(30.0)
        .get_updates_connection_pool_size(4)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)
    )

    application.add_error_handler(error_handler)

    return application