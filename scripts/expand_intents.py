import json
import random
from pathlib import Path

INPUT_PATH = Path("data/intents.json")
OUTPUT_PATH = Path("data/intents_expanded.json")

TARGET_PATTERNS = 40

# Простые шаблоны для генерации вариаций
PREFIXES = [
    "можешь",
    "пожалуйста",
    "подскажи",
    "скажи",
    "расскажи",
    "хочу",
    "мне нужно",
    "мне хотелось бы",
    "интересует",
    "подбери",
]

SUFFIXES = [
    "",
    "для дома",
    "для кухни",
    "для себя",
    "на каждый день",
    "нормальную",
    "хорошую",
    "простую",
    "удобную",
]


def generate_variations(base_patterns):
    new_patterns = set(base_patterns)

    while len(new_patterns) < TARGET_PATTERNS:
        base = random.choice(base_patterns)

        prefix = random.choice(PREFIXES)
        suffix = random.choice(SUFFIXES)

        new_variant = f"{prefix} {base} {suffix}".strip()

        # убираем двойные пробелы
        new_variant = " ".join(new_variant.split())

        new_patterns.add(new_variant)

    return list(new_patterns)


def expand_intents():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for intent in data["intents"]:
        patterns = intent["patterns"]

        if len(patterns) >= TARGET_PATTERNS:
            continue

        expanded = generate_variations(patterns)
        intent["patterns"] = expanded

        print(f"{intent['tag']}: {len(patterns)} → {len(expanded)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\nГотово! Сохранено в:", OUTPUT_PATH)


if __name__ == "__main__":
    expand_intents()