from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    INTENTS_EXPANDED_PATH,
    LABEL_ENCODER_PATH,
    LOGISTIC_REGRESSION_MAX_ITER,
    MODEL_PATH,
    MODEL_RANDOM_STATE,
    MODEL_TRAIN_TEST_SIZE,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    VECTORIZER_PATH,
)
from loaders.data_loader import load_intents
from ml.preprocessing import preprocess_text


def build_dataset(intents_data: dict) -> tuple[list[str], list[str]]:
    """
    Преобразует intents.json в список текстов и список меток.
    """
    texts: list[str] = []
    labels: list[str] = []

    for intent in intents_data["intents"]:
        tag = intent["tag"]
        patterns = intent["patterns"]

        for pattern in patterns:
            processed = preprocess_text(pattern)

            if processed:
                texts.append(processed)
                labels.append(tag)

    return texts, labels


def ensure_artifacts_dir(file_paths: list[Path]) -> None:
    """
    Создаёт родительские директории для файлов артефактов, если их ещё нет.
    """
    for file_path in file_paths:
        file_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Loading intents...")
    intents_data = load_intents(INTENTS_EXPANDED_PATH)

    texts, labels = build_dataset(intents_data)

    if not texts or not labels:
        raise ValueError("Dataset is empty. Check intents.json")

    print(f"Total samples: {len(texts)}")
    print(f"Unique intents: {len(set(labels))}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=MODEL_TRAIN_TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=LOGISTIC_REGRESSION_MAX_ITER,
        random_state=MODEL_RANDOM_STATE,
    )

    print("Training model...")
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    ensure_artifacts_dir([MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH])

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print("Artifacts saved:")
    print(f"Model: {MODEL_PATH}")
    print(f"Vectorizer: {VECTORIZER_PATH}")
    print(f"Label encoder: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
