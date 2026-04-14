import json
import re
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_PATH = "data/smalltalk_dataset.json"
MAX_PAIRS = 25000


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_valid_text(text: str) -> bool:
    """
    Базовая проверка:
    - строка
    - разумная длина
    - не пустая
    """
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
    """
    Разрешаем отдельные простые штуки типа ok,
    но длинные латинские куски считаем мусором.
    """
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
        # токсичность / грубость
        "дурак",
        "идиот",
        "тупой",
        "придур",
        "дебил",
        "заткнись",
        "пошел",
        "пошёл",
        "нах",
        "хрен",
        "черт",
        "чёрт",

        # слишком странные / нежелательные темы
        "убить",
        "убий",
        "смерт",
        "похорон",
        "наркот",
        "секс",
        "эрот",
        "порно",
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
        "пизд",
        "хер",
        "хрен",
        "сука",
        "бля",
        "бляд",
        "долбо",
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
        "красавчик",
        "письк",
        "хуй",
        "дроч",
        "занюх",
        "член",
        "кремль",
        "кремлёвск"
    ]

    return any(keyword in lowered for keyword in banned_keywords)


def looks_like_bot_friendly_text(text: str) -> bool:
    """
    Фильтруем слишком уж 'человечески-странные' реплики.
    """
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

    # слишком много капса
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if uppercase_ratio > 0.4:
            return False

    return True


def ends_reasonably(text: str) -> bool:
    """
    Отбрасываем совсем обрывочные ответы.
    """
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
    """
    Общая фильтрация одного текста.
    """
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
    """
    Фильтрация пары вопрос-ответ именно под свободный диалог бота.
    """
    q = normalize_spaces(question)
    a = normalize_spaces(answer)

    q_low = q.lower()
    a_low = a.lower()

    # убираем пары с явной бессмыслицей или слишком бытовой дикостью
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

    # ответ не должен быть намного длиннее вопроса
    q_len = len(q.split())
    a_len = len(a.split())

    if a_len > q_len * 3 + 4:
        return False

    # убираем пары, где вопрос и ответ почти одинаковы
    if q_low == a_low:
        return False

    # убираем странные обращения на "ты" в лоб, которые звучат токсично или слишком лично
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

def beautify_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return text

    # Заглавная буква в начале текста
    text = text[0].upper() + text[1:]

    # Заглавная после . ! ?
    def repl(match: re.Match) -> str:
        punctuation = match.group(1)
        letter = match.group(2).upper()
        return f"{punctuation} {letter}"

    text = re.sub(r"([.!?])\s+([а-яёa-z])", repl, text, flags=re.IGNORECASE)

    return text


def main() -> None:
    dataset = load_dataset("Den4ikAI/russian_dialogues", split="train")

    pairs: list[dict[str, str]] = []

    for item in tqdm(dataset):
        question = item.get("question", "")
        answer = item.get("answer", "")
        relevance = item.get("relevance", 0)

        if relevance != 1:
            continue

        if not is_clean_text(question):
            continue
        if not is_clean_text(answer):
            continue
        if not is_reasonable_smalltalk_pair(question, answer):
            continue

        pairs.append({
            "question": beautify_text(normalize_spaces(question)),
            "answer": beautify_text(normalize_spaces(answer)),
        })

        if len(pairs) >= MAX_PAIRS * 2:
            break

    unique_pairs = deduplicate_pairs(pairs)
    unique_pairs = unique_pairs[:MAX_PAIRS]

    result = {
        "pairs": unique_pairs
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"Собрано сырых пар: {len(pairs)}")
    print(f"Собрано уникальных пар: {len(unique_pairs)}")
    print(f"Сохранено в {OUTPUT_PATH}")

    print("\nПервые 20 пар:")
    for pair in unique_pairs[:20]:
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer']}")
        print("-" * 50)


if __name__ == "__main__":
    main()