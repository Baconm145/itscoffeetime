from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml.preprocessing import preprocess_text


class SmalltalkRetrieverError(Exception):
    """Базовая ошибка smalltalk retriever."""


class SmalltalkRetriever:
    """
    Простой retrieval-модуль для свободного общения.
    Ищет наиболее похожую реплику пользователя в smalltalk_dataset.json
    и возвращает соответствующий ответ.
    """

    def __init__(self, dataset: dict[str, Any]) -> None:
        self.dataset = dataset
        self.pairs = dataset.get("pairs", [])

        self.questions: list[str] = []
        self.answers: list[str] = []
        self.processed_questions: list[str] = []

        self._prepare_dataset()

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)

    def _prepare_dataset(self) -> None:
        if not isinstance(self.pairs, list) or not self.pairs:
            raise SmalltalkRetrieverError("smalltalk_dataset.json пустой или имеет неверную структуру.")

        for index, pair in enumerate(self.pairs):
            if not isinstance(pair, dict):
                raise SmalltalkRetrieverError(f"Элемент pairs[{index}] должен быть объектом.")

            question = pair.get("question")
            answer = pair.get("answer")

            if not isinstance(question, str) or not isinstance(answer, str):
                raise SmalltalkRetrieverError(
                    f"Элемент pairs[{index}] должен содержать строковые 'question' и 'answer'."
                )

            processed_question = preprocess_text(question)

            if processed_question:
                self.questions.append(question)
                self.answers.append(answer)
                self.processed_questions.append(processed_question)

        if not self.processed_questions:
            raise SmalltalkRetrieverError("После предобработки не осталось валидных smalltalk-вопросов.")

    def get_reply(self, text: str) -> dict[str, Any]:
        """
        Возвращает наиболее подходящий smalltalk-ответ.

        Формат:
        {
            "reply": "...",
            "matched_question": "...",
            "score": 0.0
        }
        """
        processed_text = preprocess_text(text)

        if not processed_text:
            return {
                "reply": "Можешь написать чуть подробнее?",
                "matched_question": "",
                "score": 0.0,
            }

        text_vector = self.vectorizer.transform([processed_text])
        similarities = cosine_similarity(text_vector, self.question_vectors)[0]

        best_index = similarities.argmax()
        best_score = float(similarities[best_index])

        return {
            "reply": self.answers[best_index],
            "matched_question": self.questions[best_index],
            "score": best_score,
        }

    @staticmethod
    def should_offer_product_transition(text: str) -> bool:
        """
        Простая эвристика:
        ловим смысловые сигналы, после которых уместно предложить помощь с кофемашиной.
        """
        processed = preprocess_text(text)

        trigger_keywords = {
            "кофе",
            "капучино",
            "латте",
            "эспрессо",
            "бодрить",
            "взбодрить",
            "устать",
            "устал",
            "утро",
            "проснуться",
            "не выспаться",
            "напиток",
        }

        return any(keyword in processed for keyword in trigger_keywords)