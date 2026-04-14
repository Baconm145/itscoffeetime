import re

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

# Инициализация один раз (ВАЖНО!)
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


def normalize_text(text: str) -> str:
    """
    Базовая очистка текста:
    - lowercase
    - удаление лишних символов
    - нормализация пробелов
    """
    text = text.lower().strip()

    text = re.sub(r"[^a-zа-яё0-9\s\-]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def lemmatize_text(text: str) -> str:
    """
    Лемматизация через Natasha.
    """
    doc = Doc(text)

    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    lemmas = []

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma)

    return " ".join(lemmas)


def preprocess_text(text: str) -> str:
    """
    Полная предобработка:
    1. Очистка
    2. Лемматизация
    """
    text = normalize_text(text)

    if not text:
        return text

    text = lemmatize_text(text)

    return text