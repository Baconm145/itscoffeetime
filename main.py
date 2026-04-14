from bot.telegram_bot import create_application
from config import (
    BOT_TOKEN,
    CONFIDENCE_THRESHOLD,
    DIALOGUES_PATH,
    DIALOGUE_TEXTS_PATH,
    INTENTS_PATH,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    PRODUCTS_PATH,
    ROUTES_PATH,
    SMALLTALK_DATASET_PATH,
    VECTORIZER_PATH,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    TTS_VOICE
)
from core import voice_service
from core.dialogue_manager import DialogueManager
from core.recommender import Recommender
from core.response_builder import ResponseBuilder
from core.route_engine import RouteEngine
from core.smalltalk_retriever import SmalltalkRetriever
from core.text_repository import TextRepository
from loaders.data_loader import load_all_data
from ml.intent_classifier import IntentClassifier
from core.voice_service import VoiceService


def build_dialogue_manager() -> DialogueManager:
    data = load_all_data(
        intents_path=INTENTS_PATH,
        products_path=PRODUCTS_PATH,
        routes_path=ROUTES_PATH,
        dialogues_path=DIALOGUES_PATH,
        smalltalk_dataset_path=SMALLTALK_DATASET_PATH,
        dialogue_texts_path=DIALOGUE_TEXTS_PATH,
    )

    classifier = IntentClassifier(
        model_path=MODEL_PATH,
        vectorizer_path=VECTORIZER_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
    )

    route_engine = RouteEngine(data["routes"])
    recommender = Recommender(data["products"])
    response_builder = ResponseBuilder(
        intents_data=data["intents"],
        routes_data=data["routes"],
    )
    smalltalk_retriever = SmalltalkRetriever(data["smalltalk"])
    text_repository = TextRepository(data["dialogue_texts"])

    dialogue_manager = DialogueManager(
        classifier=classifier,
        route_engine=route_engine,
        recommender=recommender,
        response_builder=response_builder,
        smalltalk_retriever=smalltalk_retriever,
        text_repository=text_repository,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    return dialogue_manager

def build_voice_service() -> VoiceService:
    return VoiceService(
        whisper_model_size=WHISPER_MODEL_SIZE,
        whisper_device=WHISPER_DEVICE,
        whisper_compute_type=WHISPER_COMPUTE_TYPE,
        tts_voice=TTS_VOICE,
    )


def main() -> None:
    dialogue_manager = build_dialogue_manager()
    vs = build_voice_service()
    application = create_application(BOT_TOKEN, dialogue_manager, vs)

    print("Telegram bot is running...")
    application.run_polling()


if __name__ == "__main__":
    main()