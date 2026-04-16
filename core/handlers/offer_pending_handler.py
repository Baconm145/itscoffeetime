from __future__ import annotations

from typing import Any

from core.response_builder import ResponseBuilder
from core.smalltalk_retriever import SmalltalkRetriever
from core.text_repository import TextRepository
from config import DEFAULT_FOLLOWUP_KEY


class OfferPendingHandler:
    """
    Обработчик режима offer_pending.
    """

    def __init__(
        self,
        smalltalk_retriever: SmalltalkRetriever,
        response_builder: ResponseBuilder,
        text_repository: TextRepository,
        decline_offer_cooldown: int,
        detect_offer_reply_fn,
        ensure_sentence_ending_fn,
        free_chat_handle_fn,
    ) -> None:
        self.smalltalk_retriever = smalltalk_retriever
        self.response_builder = response_builder
        self.text_repository = text_repository
        self.decline_offer_cooldown = decline_offer_cooldown
        self._detect_offer_reply = detect_offer_reply_fn
        self._ensure_sentence_ending = ensure_sentence_ending_fn
        self._free_chat_handle = free_chat_handle_fn

    def handle(
        self,
        text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        offer_reply = self._detect_offer_reply(text)

        if offer_reply == "accept":
            context["mode"] = "product_flow"
            context["offer_pending"] = False
            context["ad_flow_active"] = True
            context["coffee_probe_pending"] = False
            context["offer_cooldown"] = 0

            reply = self.text_repository.get("product_flow.start")
            if not reply:
                reply = self.response_builder.build_followup_question(DEFAULT_FOLLOWUP_KEY)

            return {
                "reply": reply,
                "intent": "ad_accept",
                "confidence": 1.0,
                "context": context,
            }

        if offer_reply == "decline":
            context["mode"] = "free_chat"
            context["offer_pending"] = False
            context["ad_flow_active"] = False
            context["offer_cooldown"] = self.decline_offer_cooldown
            context["coffee_probe_pending"] = False

            smalltalk_result = self.smalltalk_retriever.get_reply(
                text,
                fallback_reply="Хорошо, продолжим обычный разговор.",
            )

            prefix = self.text_repository.get("free_chat.offer_decline_prefix")
            if not prefix:
                prefix = "Хорошо, без проблем. Тогда можем просто продолжить разговор."

            reply = f"{self._ensure_sentence_ending(prefix)} {smalltalk_result['reply']}"

            return {
                "reply": reply,
                "intent": "ad_decline",
                "confidence": 1.0,
                "context": context,
            }

        context["mode"] = "free_chat"
        context["offer_pending"] = False
        return self._free_chat_handle(text, context)
