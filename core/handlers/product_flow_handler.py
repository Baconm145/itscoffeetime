from __future__ import annotations

from typing import Any

from core.recommender import ProductNotFoundError, Recommender
from core.response_builder import ResponseBuilder
from core.route_engine import RouteEngine
from core.text_repository import TextRepository
from ml.intent_classifier import IntentClassifier
from config import (
    DEFAULT_AD_TEXT_KEY,
    DEFAULT_FOLLOWUP_KEY,
    DEFAULT_OBJECTION_AD_TEXT_KEY,
    DEFAULT_TARGET_PRODUCT_FOLLOWUP_KEY,
    FOLLOWUP_PREFERENCE_INTENT_THRESHOLD,
)


class ProductFlowHandler:
    """
    Обработчик режима product_flow.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        route_engine: RouteEngine,
        recommender: Recommender,
        response_builder: ResponseBuilder,
        text_repository: TextRepository,
        confidence_threshold: float,
        decline_offer_cooldown: int,
        detect_offer_reply_fn,
        followup_preference_intent_threshold: float = FOLLOWUP_PREFERENCE_INTENT_THRESHOLD,
    ) -> None:
        self.classifier = classifier
        self.route_engine = route_engine
        self.recommender = recommender
        self.response_builder = response_builder
        self.text_repository = text_repository
        self.confidence_threshold = confidence_threshold
        self.decline_offer_cooldown = decline_offer_cooldown
        self.followup_preference_intent_threshold = followup_preference_intent_threshold
        self._detect_offer_reply = detect_offer_reply_fn

    def handle(
        self,
        text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])

        followup_intent = self._detect_followup_preference_intent(text)
        if followup_intent is not None:
            intent = followup_intent
            confidence = 1.0

        if self._detect_offer_reply(text) == "decline" or intent in {"ad_decline", "objection_no_need"}:
            context["mode"] = "free_chat"
            context["offer_pending"] = False
            context["ad_flow_active"] = False
            context["offer_cooldown"] = self.decline_offer_cooldown

            reply = self.text_repository.get("product_flow.exit_to_free_chat")
            if not reply:
                reply = (
                    "Хорошо, не будем сейчас обсуждать кофемашины. "
                    "Можем просто пообщаться дальше."
                )

            return {
                "reply": reply,
                "intent": "ad_decline",
                "confidence": 1.0,
                "context": context,
            }

        if confidence < self.confidence_threshold:
            reply = self.text_repository.get("product_flow.fallback")
            if not reply:
                reply = (
                    "Не до конца понял запрос. Могу помочь подобрать кофемашину, "
                    "сравнить модели, рассказать о характеристиках или о цене."
                )

            return {
                "reply": reply,
                "intent": "fallback",
                "confidence": confidence,
                "context": context,
            }

        reply = self.execute_product_action(intent=intent, context=context)
        context["last_intent"] = intent
        context["last_confidence"] = confidence

        return {
            "reply": reply,
            "intent": intent,
            "confidence": confidence,
            "context": context,
        }

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

    def execute_product_action(
        self,
        intent: str,
        context: dict[str, Any],
    ) -> str:
        resolved = self.route_engine.resolve(intent)
        route = resolved["route"]
        scenario_template = resolved["scenario_template"]
        global_rules = resolved["global_rules"]

        action = route.get("action")
        scenario = route.get("scenario")

        try:
            if action == "reply":
                self._apply_context_updates(route, context, intent)
                return self._build_route_reply(scenario)

            if action == "ask_followup":
                self._apply_context_updates(route, context, intent)
                question_key = scenario_template.get("question_key") or route.get("followup_type")
                return self.response_builder.build_followup_question(question_key)

            if action == "recommend_product":
                selector = route.get("product_selector")
                product = self.recommender.select_product(selector)

                if route.get("save_as_current_product"):
                    context["current_product_id"] = product["id"]

                self._apply_context_updates(route, context, intent)
                ad_text_key = scenario_template.get("ad_text_key", DEFAULT_AD_TEXT_KEY)
                return self.response_builder.build_product_recommendation(
                    product=product,
                    ad_text_key=ad_text_key,
                )

            if action == "compare_products":
                products = self.recommender.select_products_for_comparison(
                    default_products=route.get("default_products")
                )
                self._apply_context_updates(route, context, intent)
                return self.response_builder.build_comparison(products)

            if action == "recommend_alternative":
                current_product_id = context.get("current_product_id")
                if not current_product_id:
                    return self.text_repository.get("product_flow.no_current_product") or (
                        "Сначала давай выберем модель, а потом я смогу предложить альтернативу."
                    )

                current_product = self.recommender.get_product_by_id(current_product_id)
                current_budget_tier = current_product.get("budget_tier")
                alternative_rule = route.get("alternative_rule")

                if alternative_rule == "lower_budget" and current_budget_tier == "budget":
                    self._apply_context_updates(route, context, intent)
                    text = self.text_repository.get(
                        "product_flow.budget_limit_reached",
                        product_name=current_product["name"],
                    )
                    if not text:
                        text = (
                            f"{current_product['name']} уже относится к самому доступному сегменту. "
                            f"Если хочешь, я могу кратко сравнить её с более удобными моделями "
                            f"или помочь выбрать компромисс по функциям."
                        )
                    return text

                alternative_product = self.recommender.find_alternative(
                    current_product_id=current_product_id,
                    alternative_rule=alternative_rule,
                    global_rules=global_rules,
                )

                if route.get("save_as_current_product"):
                    context["current_product_id"] = alternative_product["id"]

                self._apply_context_updates(route, context, intent)
                ad_text_key = scenario_template.get("ad_text_key", DEFAULT_OBJECTION_AD_TEXT_KEY)
                return self.response_builder.build_alternative_recommendation(
                    product=alternative_product,
                    ad_text_key=ad_text_key,
                )

            if action == "describe_current_product":
                current_product_id = context.get("current_product_id")
                if not current_product_id:
                    if route.get("fallback_action") == "ask_followup":
                        question_key = route.get("followup_type", DEFAULT_TARGET_PRODUCT_FOLLOWUP_KEY)
                        return self.response_builder.build_followup_question(question_key)

                    return self.text_repository.get("product_flow.no_current_product") or (
                        "Сначала уточни, о какой модели идёт речь."
                    )

                current_product = self.recommender.get_product_by_id(current_product_id)
                self._apply_context_updates(route, context, intent)

                if scenario == "feature_description":
                    return self.response_builder.build_product_features(current_product)

                if scenario == "price_description":
                    return self.response_builder.build_product_price_description(current_product)

                return "Могу рассказать об этой модели подробнее."

            if action == "resume_or_start_recommendation":
                self._apply_context_updates(route, context, intent)
                return self.text_repository.get("product_flow.start") or (
                    self.response_builder.build_followup_question(DEFAULT_FOLLOWUP_KEY)
                )

            if action == "reply_or_compare":
                self._apply_context_updates(route, context, intent)
                if context.get("current_product_id"):
                    current_product = self.recommender.get_product_by_id(context["current_product_id"])
                    return (
                        f"Если сомневаешься, могу кратко сравнить варианты. "
                        f"Пока основной вариант - {current_product['name']}."
                    )

                return self.text_repository.get("product_flow.doubt_reply") or (
                    "Если хочешь, могу кратко сравнить варианты и подсказать, "
                    "какой выглядит более удачным."
                )

            if action == "finish_product_flow":
                self._apply_context_updates(route, context, intent)
                context["mode"] = "free_chat"
                context["offer_pending"] = False
                context["ad_flow_active"] = False
                context["offer_cooldown"] = 0

                return self.text_repository.get("product_flow.choice_done") or (
                    "Отлично, выбор сделан. Если понадобится, могу ещё что-то подсказать по моделям."
                )

            return self.text_repository.get("product_flow.fallback") or (
                self.response_builder.build_fallback_response()
            )
        except ProductNotFoundError:
            return self.text_repository.get("product_flow.fallback") or (
                self.response_builder.build_fallback_response()
            )

    def _build_route_reply(self, scenario: str | None) -> str:
        if scenario in {"exit_ad_flow", "soft_exit_from_sales"}:
            return self.text_repository.get("product_flow.exit_to_free_chat") or (
                "Хорошо, не будем сейчас обсуждать кофемашины."
            )

        if scenario == "choice_done":
            return self.text_repository.get("product_flow.choice_done") or (
                "Отлично, выбор сделан."
            )

        return self.text_repository.get("product_flow.fallback") or (
            self.response_builder.build_fallback_response()
        )

    @staticmethod
    def _apply_context_updates(
        route: dict[str, Any],
        context: dict[str, Any],
        intent: str,
    ) -> None:
        updates = route.get("set_context", {})
        if isinstance(updates, dict):
            context.update(updates)

        context["last_route_intent"] = intent
