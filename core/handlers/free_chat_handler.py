from __future__ import annotations

from typing import Any

from core.smalltalk_retriever import SmalltalkRetriever
from core.text_repository import TextRepository
from ml.intent_classifier import IntentClassifier
from config import (
    ENTRY_INTENT_THRESHOLD,
    FREE_CHAT_MIN_TURNS_FOR_PROBE,
    FREE_CHAT_NEGATIVE_COFFEE_COOLDOWN,
    FREE_CHAT_NEUTRAL_COFFEE_COOLDOWN,
    FREE_CHAT_PROBE_REPEAT_WINDOW,
)


class FreeChatHandler:
    """
    Обработчик режима free_chat.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        smalltalk_retriever: SmalltalkRetriever,
        text_repository: TextRepository,
        confidence_threshold: float,
        ensure_sentence_ending_fn,
        execute_product_action_fn,
        entry_intent_threshold: float = ENTRY_INTENT_THRESHOLD,
        coffee_signal_intent_threshold: float = 0.0,
    ) -> None:
        self.classifier = classifier
        self.smalltalk_retriever = smalltalk_retriever
        self.text_repository = text_repository
        self.confidence_threshold = confidence_threshold
        self.entry_intent_threshold = entry_intent_threshold
        self.coffee_signal_intent_threshold = coffee_signal_intent_threshold
        self._ensure_sentence_ending = ensure_sentence_ending_fn
        self._execute_product_action = execute_product_action_fn

    def handle(
        self,
        text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self._tick_cooldowns(context)

        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])

        if self._is_explicit_product_entry(intent, confidence):
            context["mode"] = "product_flow"
            context["offer_pending"] = False
            context["ad_flow_active"] = True
            context["coffee_probe_pending"] = False

            reply = self._execute_product_action(intent=intent, context=context)
            context["last_intent"] = intent
            context["last_confidence"] = confidence

            return {
                "reply": reply,
                "intent": intent,
                "confidence": confidence,
                "context": context,
            }

        smalltalk_result = self.smalltalk_retriever.get_reply(
            text,
            fallback_reply="Могу поддержать разговор. Расскажи, что у тебя сейчас на уме.",
        )
        reply = smalltalk_result["reply"]

        context["free_chat_turns"] += 1
        context["last_intent"] = "smalltalk"
        context["last_confidence"] = float(smalltalk_result["score"])

        topic_signal = self._detect_coffee_topic_signal(text)
        if topic_signal == "positive":
            return self._enter_offer_pending(context)

        if context.get("coffee_probe_pending"):
            probe_reply = self._detect_coffee_probe_reply(text)

            if probe_reply == "positive":
                return self._enter_offer_pending(context)

            if probe_reply == "negative":
                context["coffee_probe_pending"] = False
                context["coffee_cooldown"] = FREE_CHAT_NEGATIVE_COFFEE_COOLDOWN

                coffee_text = self.text_repository.get("free_chat.coffee_negative_replies")
                if coffee_text:
                    reply = coffee_text

                return {
                    "reply": reply,
                    "intent": "smalltalk",
                    "confidence": float(smalltalk_result["score"]),
                    "context": context,
                }

            context["coffee_probe_pending"] = False
            context["coffee_cooldown"] = FREE_CHAT_NEUTRAL_COFFEE_COOLDOWN

        if self._should_make_coffee_probe(context):
            probe_text = self.text_repository.get("free_chat.coffee_probes")
            if probe_text:
                reply = f"{self._ensure_sentence_ending(reply)} {probe_text}"
                context["coffee_probe_pending"] = True
                context["last_coffee_probe_turn"] = context["free_chat_turns"]

        return {
            "reply": reply,
            "intent": "smalltalk",
            "confidence": float(smalltalk_result["score"]),
            "context": context,
        }

    def _is_explicit_product_entry(
        self,
        intent: str,
        confidence: float,
    ) -> bool:
        if confidence < max(self.confidence_threshold, self.entry_intent_threshold):
            return False

        allowed_entry_intents = {
            "ask_recommendation",
        }

        return intent in allowed_entry_intents

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

    def _enter_offer_pending(self, context: dict[str, Any]) -> dict[str, Any]:
        context["mode"] = "offer_pending"
        context["offer_pending"] = True
        context["coffee_probe_pending"] = False
        context["last_offer_topic"] = "coffee_machine"
        context["offers_shown_count"] += 1

        reply = self.text_repository.get("free_chat.offer_transition")
        if not reply:
            reply = "Если хочешь, я могу помочь подобрать кофемашину под твои предпочтения."

        return {
            "reply": reply,
            "intent": "coffee_offer",
            "confidence": 1.0,
            "context": context,
        }
