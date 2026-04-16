import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from tqdm import tqdm

from config import DATA_DIR, SMALLTALK_DATASET_MAX_PAIRS


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


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_valid_text(text: str) -> bool:
    if not isinstance(text, str):
        return False

    text = normalize_spaces(text)

    if len(text) < 2:
        return False

    word_count = len(text.split())
    if word_count < 1 or word_count > 16:
        return False

    if len(text) > 120:
        return False

    return True


def has_too_many_punctuation_marks(text: str) -> bool:
    punct_count = sum(1 for ch in text if ch in "!?.,:;()-—\"'")
    return punct_count > max(8, len(text) // 4)


def has_weird_symbols(text: str) -> bool:
    return bool(re.search(r"[{}[\]<>_=+*/\\|~^№$%]", text))


def has_too_many_digits(text: str) -> bool:
    digit_count = sum(ch.isdigit() for ch in text)
    return digit_count > 3


def contains_latin_words(text: str) -> bool:
    latin_words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    return len(latin_words) > 0


def contains_bad_fragments(text: str) -> bool:
    lowered = text.lower()

    bad_fragments = [
        "http",
        "https",
        "www.",
        ".ru",
        ".com",
        "telegram",
        "t.me",
        "vk.com",
        "@",
        "#",
        "youtube",
        "rutube",
        "instagram",
        "facebook",
        "tik tok",
        "tiktok",
    ]

    return any(fragment in lowered for fragment in bad_fragments)


def contains_obviously_inappropriate_content(text: str) -> bool:
    lowered = text.lower()

    banned_keywords = [
        "дурак",
        "идиот",
        "тупой",
        "придур",
        "дебил",
        "заткнись",
        "пошел",
        "пошёл",
        "черт",
        "чёрт",
        "убить",
        "убий",
        "смерт",
        "похорон",
        "политик",
        "президент",
        "религ",
        "церков",
        "войн",
        "бомб",
        "документац",
        "накладн",
        "расходн",
        "приходн",
        "комисси",
        "идиот",
        "придур",
        "тупиц",
        "алкаш",
        "алкогол",
        "спирт",
        "подруга",
        "подружка",
        "милый",
        "милая",
        "дорогая",
        "дорогой",
        "красавица",
        "красавчик"
    ]

    return any(keyword in lowered for keyword in banned_keywords)


def looks_like_bot_friendly_text(text: str) -> bool:
    lowered = text.lower().strip()

    banned_starts = [
        "а я",
        "я просто",
        "я ржу",
        "я в шоке",
        "я хз",
        "ну ты",
        "ого, ты",
        "ты че",
        "ты чё",
        "отo",
        "ото",
    ]

    if any(lowered.startswith(prefix) for prefix in banned_starts):
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if uppercase_ratio > 0.4:
            return False

    return True


def ends_reasonably(text: str) -> bool:
    lowered = text.lower().strip()

    bad_exact_answers = {
        "ага",
        "угу",
        "ясно",
        "понятно",
        "бывает",
        "может быть",
        "не знаю",
        "хз",
        "лол",
        "ок",
        "окей",
    }

    if lowered in bad_exact_answers:
        return False

    return True


def is_clean_text(text: str) -> bool:
    if not is_valid_text(text):
        return False

    text = normalize_spaces(text)

    if has_too_many_punctuation_marks(text):
        return False
    if has_weird_symbols(text):
        return False
    if has_too_many_digits(text):
        return False
    if contains_latin_words(text):
        return False
    if contains_bad_fragments(text):
        return False
    if contains_obviously_inappropriate_content(text):
        return False
    if not looks_like_bot_friendly_text(text):
        return False
    if not ends_reasonably(text):
        return False

    return True


def is_reasonable_smalltalk_pair(question: str, answer: str) -> bool:
    q = normalize_spaces(question)
    a = normalize_spaces(answer)

    q_low = q.lower()
    a_low = a.lower()

    banned_pair_keywords = [
        "сми",
        "комисси",
        "документац",
        "ряженк",
        "кефир",
        "накладн",
        "приходн",
        "расходн",
    ]

    if any(word in q_low for word in banned_pair_keywords):
        return False
    if any(word in a_low for word in banned_pair_keywords):
        return False

    q_len = len(q.split())
    a_len = len(a.split())

    if a_len > q_len * 3 + 4:
        return False

    if q_low == a_low:
        return False

    suspicious_answer_patterns = [
        r"^ты\s+ч[её]\b",
        r"^ну\s+ты\b",
        r"^ого,\s*ты\b",
        r"^я\s+рж",
        r"^это\s+ты\b",
        r"^я\s+прозяб",
        r"^я\s+прозе",
        r"^я\s+про",
    ]

    for pattern in suspicious_answer_patterns:
        if re.search(pattern, a_low):
            return False

    return True


def beautify_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return text

    text = text[0].upper() + text[1:]

    def repl(match: re.Match) -> str:
        punctuation = match.group(1)
        letter = match.group(2).upper()
        return f"{punctuation} {letter}"

    text = re.sub(r"([.!?])\s+([а-яёa-z])", repl, text, flags=re.IGNORECASE)

    return text


def deduplicate_pairs(pairs: Iterable[dict]) -> list[dict]:
    unique_map: dict[tuple[str, str], dict] = {}
    for pair in pairs:
        key = (
            normalize_spaces(pair["question"]).lower(),
            normalize_spaces(pair["answer"]).lower(),
        )
        if key not in unique_map:
            unique_map[key] = {
                "question": beautify_text(normalize_spaces(pair["question"])),
                "answer": beautify_text(normalize_spaces(pair["answer"])),
            }
    return list(unique_map.values())


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
