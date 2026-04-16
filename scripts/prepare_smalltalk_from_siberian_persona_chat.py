import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from tqdm import tqdm

from config import DATA_DIR, SMALLTALK_DATASET_MAX_PAIRS
from scripts.prepare_smalltalk_dataset import (
    beautify_text,
    deduplicate_pairs,
    is_clean_text,
    is_reasonable_smalltalk_pair,
    normalize_spaces,
)


DEFAULT_OUTPUT_PATH = DATA_DIR / "smalltalk_dataset_siberian_persona_chat.json"
DEFAULT_ALLOWED_NAMES = ("dialog_personal_context",)
POPULAR_RUSSIAN_NAMES_TOP_30 = (
    "Александр",
    "Андрей",
    "Анна",
    "Анастасия",
    "Арина",
    "Артем",
    "Алиса",
    "Александра",
    "Василиса",
    "Вероника",
    "Виктория",
    "Даниил",
    "Дмитрий",
    "Ева",
    "Екатерина",
    "Елизавета",
    "Иван",
    "Илья",
    "Ксения",
    "Лев",
    "Максим",
    "Мария",
    "Матвей",
    "Марк",
    "Мирон",
    "Михаил",
    "Полина",
    "Роман",
    "София",
    "Тимофей",
)
POPULAR_RUSSIAN_NAMES_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(name) for name in POPULAR_RUSSIAN_NAMES_TOP_30)
    + r")\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Готовит smalltalk-датасет в формате {'pairs': [{'question', 'answer'}]} "
            "из Hugging Face датасета SiberiaSoft/SiberianPersonaChat."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Куда сохранить результат. По умолчанию: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=SMALLTALK_DATASET_MAX_PAIRS,
        help=f"Максимум уникальных пар. По умолчанию: {SMALLTALK_DATASET_MAX_PAIRS}",
    )
    parser.add_argument(
        "--dataset",
        default="SiberiaSoft/SiberianPersonaChat",
        help="Имя исходного датасета на Hugging Face.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Сплит Hugging Face датасета. По умолчанию: train.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=list(DEFAULT_ALLOWED_NAMES),
        help=(
            "Какие значения колонки 'name' использовать. "
            "По умолчанию: dialog_personal_context."
        ),
    )
    parser.add_argument(
        "--disable-popular-name-filter",
        action="store_true",
        help=(
            "Отключить фильтр по 30 популярным русским именам. "
            "По умолчанию фильтр включен."
        ),
    )
    return parser.parse_args()


def extract_last_interlocutor_message(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    matches = re.findall(r"Собеседник:\s*(.*?)\s*Ты:\s*", text, flags=re.DOTALL)
    if not matches:
        return None

    candidate = normalize_spaces(matches[-1])
    if not candidate:
        return None

    return candidate


def is_smalltalk_candidate(
    question: str,
    answer: str,
    source_input: str = "",
    use_popular_name_filter: bool = True,
) -> bool:
    question = normalize_spaces(question)
    answer = normalize_spaces(answer)
    source_input = normalize_spaces(source_input)

    if not is_clean_text(question) or not is_clean_text(answer):
        return False

    if not is_reasonable_smalltalk_pair(question, answer):
        return False

    if use_popular_name_filter and not contains_popular_russian_name(source_input, question, answer):
        return False

    q_low = question.lower()
    a_low = answer.lower()

    banned_fragments = (
        "продолжи диалог",
        "составь список",
        "опиши персонажа",
        "предложи варианты",
        "собеседник:",
        "ты:",
        "законы",
        "инвестици",
        "политик",
        "религи",
        "церков",
        "президент",
        "криптовалют",
        "недвижимост",
    )
    if any(fragment in q_low for fragment in banned_fragments):
        return False
    if any(fragment in a_low for fragment in banned_fragments):
        return False

    if len(question.split()) > 14:
        return False
    if len(answer.split()) > 18:
        return False

    return True


def contains_popular_russian_name(*texts: str) -> bool:
    for text in texts:
        if isinstance(text, str) and POPULAR_RUSSIAN_NAMES_PATTERN.search(text):
            return True
    return False


def build_pairs(
    dataset_name: str,
    split: str,
    allowed_names: set[str],
    max_pairs: int,
    use_popular_name_filter: bool,
) -> list[dict[str, str]]:
    dataset = load_dataset(dataset_name, split=split)
    raw_pairs: list[dict[str, str]] = []

    for item in tqdm(dataset, desc="Converting"):
        item_name = item.get("name", "")
        if item_name not in allowed_names:
            continue

        source_input = item.get("input", "")
        question = extract_last_interlocutor_message(source_input)
        answer = item.get("output", "")

        if not question or not isinstance(answer, str):
            continue

        if not is_smalltalk_candidate(
            question,
            answer,
            source_input=source_input,
            use_popular_name_filter=use_popular_name_filter,
        ):
            continue

        raw_pairs.append(
            {
                "question": beautify_text(question),
                "answer": beautify_text(answer),
            }
        )

        if len(raw_pairs) >= max_pairs * 4:
            break

    unique_pairs = deduplicate_pairs(raw_pairs)
    return unique_pairs[:max_pairs]


def main() -> None:
    args = parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(
        dataset_name=args.dataset,
        split=args.split,
        allowed_names=set(args.names),
        max_pairs=args.max_pairs,
        use_popular_name_filter=not args.disable_popular_name_filter,
    )

    result = {"pairs": pairs}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Сохранено пар: {len(pairs)}")
    print(f"Файл: {output_path}")
    print("Первые 10 пар:")
    for pair in pairs[:10]:
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
