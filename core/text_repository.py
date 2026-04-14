import random
from typing import Any


class TextRepositoryError(Exception):
    """Базовая ошибка репозитория текстов."""


class TextRepository:
    """
    Репозиторий системных реплик бота.
    Берёт тексты из dialogue_texts.json.
    """

    def __init__(self, texts_data: dict[str, Any]) -> None:
        self.texts_data = texts_data

    def get(self, path: str, **kwargs: Any) -> str:
        """
        Возвращает случайную строку по пути вида:
        'free_chat.offer_transition'
        'product_flow.start'
        """
        value = self._get_by_path(path)

        if isinstance(value, list):
            if not value:
                return ""
            text = random.choice(value)
        elif isinstance(value, str):
            text = value
        else:
            return ""

        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass

        return text

    def _get_by_path(self, path: str) -> Any:
        parts = path.split(".")
        current: Any = self.texts_data

        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)

        return current