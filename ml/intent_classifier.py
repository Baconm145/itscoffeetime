from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.preprocessing import preprocess_text


class IntentClassifierError(Exception):
    """Базовая ошибка классификатора интентов."""


class ModelArtifactsNotFoundError(IntentClassifierError):
    """Ошибка, если не найдены файлы модели."""


class IntentClassifier:
    """
    Классификатор интентов на основе сохранённых артефактов:
    - model.pkl
    - vectorizer.pkl
    - label_encoder.pkl
    """

    def __init__(
        self,
        model_path: Path,
        vectorizer_path: Path,
        label_encoder_path: Path,
    ) -> None:
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.label_encoder_path = label_encoder_path

        self.model: Any = None
        self.vectorizer: Any = None
        self.label_encoder: Any = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Загружает модель, векторайзер и label encoder.
        """
        for path in (self.model_path, self.vectorizer_path, self.label_encoder_path):
            if not path.exists():
                raise ModelArtifactsNotFoundError(f"Файл не найден: {path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.label_encoder = joblib.load(self.label_encoder_path)

    def predict(self, text: str) -> dict[str, Any]:
        """
        Предсказывает интент для входного текста.

        Возвращает словарь:
        {
            "text": исходный текст,
            "processed_text": обработанный текст,
            "intent": предсказанный интент,
            "confidence": уверенность модели
        }
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        processed_text = preprocess_text(text)

        if not processed_text:
            return {
                "text": text,
                "processed_text": processed_text,
                "intent": "fallback",
                "confidence": 0.0,
            }

        text_vector = self.vectorizer.transform([processed_text])
        predicted_class_index = self.model.predict(text_vector)[0]
        predicted_intent = str(
            self.label_encoder.inverse_transform([predicted_class_index])[0]
        )

        confidence = self._get_confidence(text_vector, predicted_class_index)

        return {
            "text": text,
            "processed_text": processed_text,
            "intent": predicted_intent,
            "confidence": confidence,
        }

    def _get_confidence(self, text_vector: Any, predicted_class_index: int) -> float:
        """
        Возвращает уверенность модели.

        Если модель поддерживает predict_proba, используем её.
        Иначе возвращаем 1.0 для совместимости.
        """
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(text_vector)[0]
            return float(np.max(probabilities))

        return 1.0