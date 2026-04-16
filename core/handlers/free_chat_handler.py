from __future__ import annotations

from typing import Any

from core.smalltalk_retriever import SmalltalkRetriever
from core.text_repository import TextRepository
from ml.intent_classifier import IntentClassifier
from config import FREE_CHAT_NEGATIVE_COFFEE_COOLDOWN, FREE_CHAT_NEUTRAL_COFFEE_COOLDOWN


class FreeChatHandler:
    """
    Обработчик режима free_chat.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        smalltalk_retriever: SmalltalkRetriever,
        text_repository: TextRepository,
        is_explicit_product_entry_fn,
        detect_coffee_topic_signal_fn,
        detect_coffee_probe_reply_fn,
        should_make_coffee_probe_fn,
        ensure_sentence_ending_fn,
        tick_cooldowns_fn,
        execute_product_action_fn,
    ) -> None:
        self.classifier = classifier
        self.smalltalk_retriever = smalltalk_retriever
        self.text_repository = text_repository
        self._is_explicit_product_entry = is_explicit_product_entry_fn
        self._detect_coffee_topic_signal = detect_coffee_topic_signal_fn
        self._detect_coffee_probe_reply = detect_coffee_probe_reply_fn
        self._should_make_coffee_probe = should_make_coffee_probe_fn
        self._ensure_sentence_ending = ensure_sentence_ending_fn
        self._tick_cooldowns = tick_cooldowns_fn
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
                    reply = f"{self._ensure_sentence_ending(reply)} {coffee_text}"

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
