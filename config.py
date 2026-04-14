from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

INTENTS_PATH = DATA_DIR / "intents_expanded.json"
PRODUCTS_PATH = DATA_DIR / "products.json"
ROUTES_PATH = DATA_DIR / "dialogue_routes.json"
DIALOGUES_PATH = DATA_DIR / "dialogues.txt"
SMALLTALK_DATASET_PATH = DATA_DIR / "smalltalk_dataset.json"
DIALOGUE_TEXTS_PATH = DATA_DIR / "dialogue_texts.json"

ML_DIR = BASE_DIR / "ml" / "artifacts"

MODEL_PATH = ML_DIR / "model.pkl"
VECTORIZER_PATH = ML_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = ML_DIR / "label_encoder.pkl"

CONFIDENCE_THRESHOLD = 0.05

BOT_TOKEN = "8749607600:AAH9uRcl7nY-lOJO8X4l74KRxjoxcSU8fn0"