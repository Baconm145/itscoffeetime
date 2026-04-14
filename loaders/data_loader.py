import json
from pathlib import Path
from typing import Any


class DataLoaderError(Exception):
    """Базовая ошибка загрузчика данных."""


class FileNotFoundDataError(DataLoaderError):
    """Ошибка, если файл не найден."""


class InvalidJsonDataError(DataLoaderError):
    """Ошибка, если JSON поврежден или имеет неверный формат."""


class InvalidStructureDataError(DataLoaderError):
    """Ошибка, если у данных неверная структура."""


def _read_json_file(file_path: Path) -> Any:
    if not file_path.exists():
        raise FileNotFoundDataError(f"Файл не найден: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise InvalidJsonDataError(
            f"Некорректный JSON в файле: {file_path}"
        ) from exc


def _read_text_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundDataError(f"Файл не найден: {file_path}")

    return file_path.read_text(encoding="utf-8")


def load_intents(file_path: Path) -> dict[str, Any]:
    data = _read_json_file(file_path)

    if not isinstance(data, dict):
        raise InvalidStructureDataError(
            f"intents.json должен содержать JSON-объект: {file_path}"
        )

    if "intents" not in data:
        raise InvalidStructureDataError(
            f"В intents.json отсутствует ключ 'intents': {file_path}"
        )

    intents = data["intents"]

    if not isinstance(intents, list):
        raise InvalidStructureDataError(
            f"Поле 'intents' должно быть списком: {file_path}"
        )

    for index, intent in enumerate(intents):
        if not isinstance(intent, dict):
            raise InvalidStructureDataError(
                f"Элемент intents[{index}] должен быть объектом"
            )

        required_keys = {"tag", "patterns"}
        missing_keys = required_keys - intent.keys()

        if missing_keys:
            raise InvalidStructureDataError(
                f"В intents[{index}] отсутствуют ключи: {missing_keys}"
            )

        if not isinstance(intent["tag"], str):
            raise InvalidStructureDataError(
                f"Поле 'tag' в intents[{index}] должно быть строкой"
            )

        if not isinstance(intent["patterns"], list):
            raise InvalidStructureDataError(
                f"Поле 'patterns' в intents[{index}] должно быть списком"
            )

    return data


def load_products(file_path: Path) -> dict[str, Any]:
    data = _read_json_file(file_path)

    if not isinstance(data, dict):
        raise InvalidStructureDataError(
            f"products.json должен содержать JSON-объект: {file_path}"
        )

    if "products" not in data:
        raise InvalidStructureDataError(
            f"В products.json отсутствует ключ 'products': {file_path}"
        )

    products = data["products"]

    if not isinstance(products, list):
        raise InvalidStructureDataError(
            f"Поле 'products' должно быть списком: {file_path}"
        )

    for index, product in enumerate(products):
        if not isinstance(product, dict):
            raise InvalidStructureDataError(
                f"Элемент products[{index}] должен быть объектом"
            )

        required_keys = {"id", "name", "brand", "budget_tier"}
        missing_keys = required_keys - product.keys()

        if missing_keys:
            raise InvalidStructureDataError(
                f"В products[{index}] отсутствуют ключи: {missing_keys}"
            )

        if not isinstance(product["id"], str):
            raise InvalidStructureDataError(
                f"Поле 'id' в products[{index}] должно быть строкой"
            )

        if not isinstance(product["name"], str):
            raise InvalidStructureDataError(
                f"Поле 'name' в products[{index}] должно быть строкой"
            )

    return data


def load_routes(file_path: Path) -> dict[str, Any]:
    data = _read_json_file(file_path)

    if not isinstance(data, dict):
        raise InvalidStructureDataError(
            f"dialogue_routes.json должен содержать JSON-объект: {file_path}"
        )

    required_top_keys = {
        "routes",
        "scenario_templates",
        "followup_questions",
        "global_rules",
    }
    missing_keys = required_top_keys - data.keys()

    if missing_keys:
        raise InvalidStructureDataError(
            f"В dialogue_routes.json отсутствуют ключи: {missing_keys}"
        )

    if not isinstance(data["routes"], dict):
        raise InvalidStructureDataError(
            "Поле 'routes' должно быть объектом"
        )

    return data


def load_smalltalk_dataset(file_path: Path) -> dict[str, Any]:
    data = _read_json_file(file_path)

    if not isinstance(data, dict):
        raise InvalidStructureDataError(
            f"smalltalk_dataset.json должен содержать JSON-объект: {file_path}"
        )

    if "pairs" not in data:
        raise InvalidStructureDataError(
            f"В smalltalk_dataset.json отсутствует ключ 'pairs': {file_path}"
        )

    pairs = data["pairs"]

    if not isinstance(pairs, list):
        raise InvalidStructureDataError(
            f"Поле 'pairs' должно быть списком: {file_path}"
        )

    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise InvalidStructureDataError(
                f"Элемент pairs[{index}] должен быть объектом"
            )

        required_keys = {"question", "answer"}
        missing_keys = required_keys - pair.keys()

        if missing_keys:
            raise InvalidStructureDataError(
                f"В pairs[{index}] отсутствуют ключи: {missing_keys}"
            )

        if not isinstance(pair["question"], str):
            raise InvalidStructureDataError(
                f"Поле 'question' в pairs[{index}] должно быть строкой"
            )

        if not isinstance(pair["answer"], str):
            raise InvalidStructureDataError(
                f"Поле 'answer' в pairs[{index}] должно быть строкой"
            )

    return data


def load_dialogue_texts(file_path: Path) -> dict[str, Any]:
    data = _read_json_file(file_path)

    if not isinstance(data, dict):
        raise InvalidStructureDataError(
            f"dialogue_texts.json должен содержать JSON-объект: {file_path}"
        )

    return data


def load_dialogues(file_path: Path) -> str:
    return _read_text_file(file_path)


def load_all_data(
    intents_path: Path,
    products_path: Path,
    routes_path: Path,
    dialogues_path: Path,
    smalltalk_dataset_path: Path,
    dialogue_texts_path: Path,
) -> dict[str, Any]:
    intents_data = load_intents(intents_path)
    products_data = load_products(products_path)
    routes_data = load_routes(routes_path)
    dialogues_text = load_dialogues(dialogues_path)
    smalltalk_data = load_smalltalk_dataset(smalltalk_dataset_path)
    dialogue_texts_data = load_dialogue_texts(dialogue_texts_path)

    return {
        "intents": intents_data,
        "products": products_data,
        "routes": routes_data,
        "dialogues": dialogues_text,
        "smalltalk": smalltalk_data,
        "dialogue_texts": dialogue_texts_data,
    }