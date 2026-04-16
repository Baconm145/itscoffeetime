from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SMALLTALK_MIN_SIMILARITY, SMALLTALK_TFIDF_NGRAM_RANGE
from ml.preprocessing import preprocess_text


class SmalltalkRetrieverError(Exception):
    """Базовая ошибка smalltalk retriever."""


class SmalltalkRetriever:
    """
    Простой retrieval-модуль для свободного общения.
    """

    def __init__(
        self,
        dataset: dict[str, Any],
        min_similarity: float = SMALLTALK_MIN_SIMILARITY,
    ) -> None:
        self.dataset = dataset
        self.pairs = dataset.get("pairs", [])
        self.min_similarity = min_similarity

        self.questions: list[str] = []
        self.answers: list[str] = []
        self.processed_questions: list[str] = []

        self._prepare_dataset()

        self.vectorizer = TfidfVectorizer(ngram_range=SMALLTALK_TFIDF_NGRAM_RANGE)
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)

    def _prepare_dataset(self) -> None:
        if not isinstance(self.pairs, list) or not self.pairs:
            raise SmalltalkRetrieverError("Датасет пустой или имеет неверную структуру.")

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

    def get_reply(
        self,
        text: str,
        fallback_reply: str | None = None,
    ) -> dict[str, Any]:
        processed_text = preprocess_text(text)

        if not processed_text:
            return {
                "reply": fallback_reply or "Можешь написать чуть подробнее?",
                "matched_question": "",
                "score": 0.0,
            }

        text_vector = self.vectorizer.transform([processed_text])
        similarities = cosine_similarity(text_vector, self.question_vectors)[0]

        best_index = similarities.argmax()
        best_score = float(similarities[best_index])

        if best_score < self.min_similarity:
            return {
                "reply": fallback_reply or "Могу поддержать разговор. Расскажи чуть подробнее.",
                "matched_question": "",
                "score": best_score,
            }

        return {
            "reply": self.answers[best_index],
            "matched_question": self.questions[best_index],
            "score": best_score,
        }
