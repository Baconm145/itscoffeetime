from __future__ import annotations

from typing import Any

from config import (
    COFFEE_SIGNAL_INTENT_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    ENTRY_INTENT_THRESHOLD,
    FOLLOWUP_PREFERENCE_INTENT_THRESHOLD,
    FREE_CHAT_MIN_TURNS_FOR_PROBE,
    FREE_CHAT_PROBE_REPEAT_WINDOW,
    OFFER_COOLDOWN_AFTER_DECLINE,
    OFFER_REPLY_INTENT_THRESHOLD,
)
from core.handlers.free_chat_handler import FreeChatHandler
from core.handlers.offer_pending_handler import OfferPendingHandler
from core.handlers.product_flow_handler import ProductFlowHandler
from core.recommender import Recommender
from core.response_builder import ResponseBuilder
from core.route_engine import RouteEngine
from core.smalltalk_retriever import SmalltalkRetriever
from core.text_repository import TextRepository
from ml.intent_classifier import IntentClassifier


class DialogueManager:
    """
    Центральный менеджер диалога.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        route_engine: RouteEngine,
        recommender: Recommender,
        response_builder: ResponseBuilder,
        smalltalk_retriever: SmalltalkRetriever,
        text_repository: TextRepository,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        offer_cooldown_after_decline: int = OFFER_COOLDOWN_AFTER_DECLINE,
        entry_intent_threshold: float = ENTRY_INTENT_THRESHOLD,
        offer_reply_intent_threshold: float = OFFER_REPLY_INTENT_THRESHOLD,
        followup_preference_intent_threshold: float = FOLLOWUP_PREFERENCE_INTENT_THRESHOLD,
        coffee_signal_intent_threshold: float = COFFEE_SIGNAL_INTENT_THRESHOLD,
    ) -> None:
        self.classifier = classifier
        self.route_engine = route_engine
        self.recommender = recommender
        self.response_builder = response_builder
        self.smalltalk_retriever = smalltalk_retriever
        self.text_repository = text_repository
        self.confidence_threshold = confidence_threshold
        self.offer_cooldown_after_decline = offer_cooldown_after_decline
        self.entry_intent_threshold = entry_intent_threshold
        self.offer_reply_intent_threshold = offer_reply_intent_threshold
        self.followup_preference_intent_threshold = followup_preference_intent_threshold
        self.coffee_signal_intent_threshold = coffee_signal_intent_threshold

        self.product_flow_handler = ProductFlowHandler(
            classifier=self.classifier,
            route_engine=self.route_engine,
            recommender=self.recommender,
            response_builder=self.response_builder,
            text_repository=self.text_repository,
            confidence_threshold=self.confidence_threshold,
            decline_offer_cooldown=self.offer_cooldown_after_decline,
            detect_followup_preference_intent_fn=self._detect_followup_preference_intent,
            detect_offer_reply_fn=self._detect_offer_reply,
        )
        self.free_chat_handler = FreeChatHandler(
            classifier=self.classifier,
            smalltalk_retriever=self.smalltalk_retriever,
            text_repository=self.text_repository,
            is_explicit_product_entry_fn=self._is_explicit_product_entry,
            detect_coffee_topic_signal_fn=self._detect_coffee_topic_signal,
            detect_coffee_probe_reply_fn=self._detect_coffee_probe_reply,
            should_make_coffee_probe_fn=self._should_make_coffee_probe,
            ensure_sentence_ending_fn=self._ensure_sentence_ending,
            tick_cooldowns_fn=self._tick_cooldowns,
            execute_product_action_fn=self.product_flow_handler.execute_product_action,
        )
        self.offer_pending_handler = OfferPendingHandler(
            smalltalk_retriever=self.smalltalk_retriever,
            response_builder=self.response_builder,
            text_repository=self.text_repository,
            decline_offer_cooldown=self.offer_cooldown_after_decline,
            detect_offer_reply_fn=self._detect_offer_reply,
            ensure_sentence_ending_fn=self._ensure_sentence_ending,
            free_chat_handle_fn=self.free_chat_handler.handle,
        )

    def process_message(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if context is None:
            context = {}

        self._init_context(context)

        if context["mode"] == "offer_pending":
            return self.offer_pending_handler.handle(text, context)

        if context["mode"] == "product_flow":
            return self.product_flow_handler.handle(text, context)

        return self.free_chat_handler.handle(text, context)

    def _is_explicit_product_entry(
        self,
        intent: str,
        confidence: float,
    ) -> bool:
        if confidence < max(self.confidence_threshold, self.entry_intent_threshold):
            return False

        allowed_entry_intents = {
            "ask_recommendation"
        }

        return intent in allowed_entry_intents

    def _detect_offer_reply(self, text: str) -> str | None:
        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])

        if (
            confidence >= max(self.confidence_threshold, self.offer_reply_intent_threshold)
            and intent in {"ad_accept", "generic_positive"}
        ):
            return "accept"

        if (
            confidence >= max(self.confidence_threshold, self.offer_reply_intent_threshold)
            and intent in {"ad_decline", "generic_negative"}
        ):
            return "decline"

        return None

    def _detect_followup_preference_intent(self, text: str) -> str | None:
        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])

        if confidence < max(self.confidence_threshold, self.followup_preference_intent_threshold):
            return None

        if intent in {
            "ask_automatic_option",
            "ask_budget_option",
            "ask_cappuccino_option",
        }:
            return intent

        return None

    def _detect_coffee_topic_signal(self, text: str) -> str | None:
        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])

        if confidence < max(self.confidence_threshold, self.coffee_signal_intent_threshold):
            return None

        if intent == "coffee_negative":
            return "negative"

        if intent == "coffee_positive":
            return "positive"

        return None

    def _detect_coffee_probe_reply(self, text: str) -> str | None:
        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])

        if confidence < self.confidence_threshold:
            return None

        if intent in {"coffee_positive", "generic_positive"}:
            return "positive"

        if intent in {"coffee_negative", "generic_negative"}:
            return "negative"

        return None

    @staticmethod
    def _init_context(context: dict[str, Any]) -> None:
        context.setdefault("mode", "free_chat")
        context.setdefault("offer_pending", False)
        context.setdefault("ad_flow_active", False)
        context.setdefault("free_chat_turns", 0)
        context.setdefault("offers_shown_count", 0)
        context.setdefault("offer_cooldown", 0)
        context.setdefault("last_offer_topic", None)
        context.setdefault("current_product_id", None)
        context.setdefault("last_intent", None)
        context.setdefault("last_confidence", 0.0)
        context.setdefault("coffee_probe_pending", False)
        context.setdefault("coffee_cooldown", 0)
        context.setdefault("last_coffee_probe_turn", 0)

    def _should_make_coffee_probe(
        self,
        context: dict[str, Any],
    ) -> bool:
        if context.get("mode") != "free_chat":
            return False

        if context.get("offer_pending"):
            return False

        if context.get("ad_flow_active"):
            return False

        if context.get("coffee_probe_pending"):
            return False

        if context.get("coffee_cooldown", 0) > 0:
            return False

        if context.get("offer_cooldown", 0) > 0:
            return False

        if context.get("free_chat_turns", 0) < FREE_CHAT_MIN_TURNS_FOR_PROBE:
            return False

        last_probe_turn = context.get("last_coffee_probe_turn", 0)
        if (
            last_probe_turn > 0
            and context.get("free_chat_turns", 0) - last_probe_turn < FREE_CHAT_PROBE_REPEAT_WINDOW
        ):
            return False

        return True

    @staticmethod
    def _tick_cooldowns(context: dict[str, Any]) -> None:
        for key in ("coffee_cooldown", "offer_cooldown"):
            if context.get(key, 0) > 0:
                context[key] -= 1

    @staticmethod
    def _ensure_sentence_ending(text: str) -> str:
        text = text.strip()
        if not text:
            return text

        if text[-1] not in ".!?":
            text += "."
        return text
