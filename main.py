from bot.telegram_bot import create_application
from config import (
    BOT_TOKEN,
    COFFEE_SIGNAL_INTENT_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    DIALOGUE_TEXTS_PATH,
    ENTRY_INTENT_THRESHOLD,
    FOLLOWUP_PREFERENCE_INTENT_THRESHOLD,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    OFFER_COOLDOWN_AFTER_DECLINE,
    OFFER_REPLY_INTENT_THRESHOLD,
    PRODUCTS_PATH,
    ROUTES_PATH,
    SMALLTALK_MIN_SIMILARITY,
    SMALLTALK_DATASET_PATH,
    TTS_VOICE,
    VECTORIZER_PATH,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)
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
        products_path=PRODUCTS_PATH,
        routes_path=ROUTES_PATH,
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
        routes_data=data["routes"],
    )
    smalltalk_retriever = SmalltalkRetriever(
        data["smalltalk"],
        min_similarity=SMALLTALK_MIN_SIMILARITY,
    )
    text_repository = TextRepository(data["dialogue_texts"])

    dialogue_manager = DialogueManager(
        classifier=classifier,
        route_engine=route_engine,
        recommender=recommender,
        response_builder=response_builder,
        smalltalk_retriever=smalltalk_retriever,
        text_repository=text_repository,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        offer_cooldown_after_decline=OFFER_COOLDOWN_AFTER_DECLINE,
        entry_intent_threshold=ENTRY_INTENT_THRESHOLD,
        offer_reply_intent_threshold=OFFER_REPLY_INTENT_THRESHOLD,
        followup_preference_intent_threshold=FOLLOWUP_PREFERENCE_INTENT_THRESHOLD,
        coffee_signal_intent_threshold=COFFEE_SIGNAL_INTENT_THRESHOLD,
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
