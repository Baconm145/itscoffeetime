from __future__ import annotations

from typing import Any

from core.recommender import ProductNotFoundError, Recommender
from core.response_builder import ResponseBuilder
from core.route_engine import RouteEngine
from core.smalltalk_retriever import SmalltalkRetriever
from core.text_repository import TextRepository
from ml.intent_classifier import IntentClassifier
from ml.preprocessing import preprocess_text


class DialogueManager:
    """
    Центральный менеджер диалога.

    Режимы:
    - free_chat: свободное общение через smalltalk dataset
    - offer_pending: бот предложил помочь с выбором кофемашины и ждёт ответ
    - product_flow: активна товарная ветка через intents/routes/products
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        route_engine: RouteEngine,
        recommender: Recommender,
        response_builder: ResponseBuilder,
        smalltalk_retriever: SmalltalkRetriever,
        text_repository: TextRepository,
        confidence_threshold: float = 0.05,
    ) -> None:
        self.classifier = classifier
        self.route_engine = route_engine
        self.recommender = recommender
        self.response_builder = response_builder
        self.smalltalk_retriever = smalltalk_retriever
        self.text_repository = text_repository
        self.confidence_threshold = confidence_threshold

    def process_message(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if context is None:
            context = {}

        self._init_context(context)

        mode = context["mode"]

        if mode == "offer_pending":
            return self._handle_offer_pending(text, context)

        if mode == "product_flow":
            return self._handle_product_flow(text, context)

        if mode == "coffee_chat":
            return self._handle_coffee_chat(text, context)

        return self._handle_free_chat(text, context)

    def _handle_free_chat(
        self,
        text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Свободное общение.

        В product_flow переходим только если:
        1) пользователь явно сам заговорил о выборе кофемашины;
        2) или ранее бот сделал оффер, а пользователь согласился
           (это уже обрабатывается в offer_pending).
        """
        self._tick_cooldowns(context)
        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])

        if self._is_explicit_product_entry(text, intent, confidence):
            context["mode"] = "product_flow"
            context["offer_pending"] = False
            context["ad_flow_active"] = True
            context["user_declined_offer_recently"] = False
            context["bridge_pending"] = False
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

        smalltalk_result = self.smalltalk_retriever.get_reply(text)
        reply = smalltalk_result["reply"]

        context["free_chat_turns"] += 1
        context["last_intent"] = "smalltalk"
        context["last_confidence"] = float(smalltalk_result["score"])

        # Если пользователь сам заговорил о кофе — уходим в coffee_chat
        coffee_reply = self._detect_coffee_interest_reply(text)
        if coffee_reply == "positive":
            context["mode"] = "coffee_chat"
            context["coffee_probe_pending"] = False
            context["bridge_pending"] = False
            context["coffee_topic_turns"] += 1
            context["coffee_chat_turns"] = 1
            context["last_offer_topic"] = "coffee_machine"

            coffee_text = self.text_repository.get("free_chat.coffee_positive_replies")
            if coffee_text:
                reply = coffee_text

            return {
                "reply": reply,
                "intent": "coffee_interest",
                "confidence": 1.0,
                "context": context,
            }

        if coffee_reply == "negative":
            context["coffee_probe_pending"] = False
            context["coffee_cooldown"] = 6

        # Если бот сам поднял тему кофе и ждёт реакцию
        if context.get("coffee_probe_pending"):
            probe_reply = self._detect_coffee_interest_reply(text)

            if probe_reply == "positive":
                context["mode"] = "coffee_chat"
                context["coffee_probe_pending"] = False
                context["bridge_pending"] = False
                context["coffee_topic_turns"] += 1
                context["coffee_chat_turns"] = 1
                context["last_offer_topic"] = "coffee_machine"

                coffee_text = self.text_repository.get("free_chat.coffee_positive_replies")
                if coffee_text:
                    reply = coffee_text

                return {
                    "reply": reply,
                    "intent": "coffee_interest",
                    "confidence": 1.0,
                    "context": context,
                }

            if probe_reply == "negative":
                context["coffee_probe_pending"] = False
                context["coffee_cooldown"] = 6

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
            context["coffee_cooldown"] = 5

        # Иногда бот сам мягко поднимает тему кофе
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

    def _handle_coffee_chat(
            self,
            text: str,
            context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Отдельная ветка разговора о кофе.
        Здесь:
        - поддерживаем тему кофе
        - даём мостик
        - затем делаем оффер на подбор кофемашины
        """
        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])

        # Если пользователь уже явно просит подобрать кофемашину — сразу уходим в product_flow
        if self._is_explicit_product_entry(text, intent, confidence):
            context["mode"] = "product_flow"
            context["offer_pending"] = False
            context["ad_flow_active"] = True
            context["bridge_pending"] = False
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

        smalltalk_result = self.smalltalk_retriever.get_reply(text)
        reply = smalltalk_result["reply"]

        context["coffee_chat_turns"] += 1
        context["coffee_topic_turns"] += 1
        context["last_intent"] = "coffee_chat"
        context["last_confidence"] = float(smalltalk_result["score"])

        coffee_reply = self._detect_coffee_interest_reply(text)

        if coffee_reply == "negative":
            context["mode"] = "free_chat"
            context["bridge_pending"] = False
            context["coffee_probe_pending"] = False
            context["coffee_cooldown"] = 8
            context["coffee_chat_turns"] = 0

            negative_text = self.text_repository.get("free_chat.coffee_negative_replies")
            if negative_text:
                reply = negative_text

            return {
                "reply": reply,
                "intent": "smalltalk",
                "confidence": 1.0,
                "context": context,
            }

        # Сначала мостик
        if self._should_trigger_bridge(text, context):
            bridge_text = self.text_repository.get("free_chat.bridge_replies")
            if bridge_text:
                reply = bridge_text

            context["bridge_pending"] = True

            return {
                "reply": reply,
                "intent": "coffee_chat",
                "confidence": float(smalltalk_result["score"]),
                "context": context,
            }

        # Потом оффер
        if self._should_make_offer(context):
            offer_text = self.text_repository.get("free_chat.offer_transition")
            if offer_text:
                reply = offer_text

            context["mode"] = "offer_pending"
            context["offer_pending"] = True
            context["bridge_pending"] = False
            context["offers_shown_count"] += 1
            context["last_offer_topic"] = "coffee_machine"

            return {
                "reply": reply,
                "intent": "coffee_offer",
                "confidence": 1.0,
                "context": context,
            }

        return {
            "reply": reply,
            "intent": "coffee_chat",
            "confidence": float(smalltalk_result["score"]),
            "context": context,
        }

    def _handle_offer_pending(
        self,
        text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Бот уже предложил помощь с подбором кофемашины и ждёт:
        - accept -> product_flow
        - decline -> free_chat
        - anything else -> обратно free_chat
        """
        offer_reply = self._detect_offer_reply(text)

        if offer_reply == "accept":
            context["mode"] = "product_flow"
            context["offer_pending"] = False
            context["ad_flow_active"] = True
            context["user_declined_offer_recently"] = False
            context["bridge_pending"] = False
            context["coffee_probe_pending"] = False
            context["coffee_chat_turns"] = 0

            reply = self.text_repository.get("product_flow.start")
            if not reply:
                reply = self.response_builder.build_followup_question("budget_or_usage")

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
            context["user_declined_offer_recently"] = True
            context["bridge_pending"] = False
            context["offer_cooldown"] = 0
            context["coffee_probe_pending"] = False
            context["coffee_chat_turns"] = 0

            smalltalk_result = self.smalltalk_retriever.get_reply(text)

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

        return self._handle_free_chat(text, context)

    def _handle_product_flow(
        self,
        text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Товарная ветка:
        всё идёт через classifier -> route_engine -> recommender/response_builder.
        """
        prediction = self.classifier.predict(text)
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])
        followup_intent = self._detect_followup_preference_intent(text)
        if followup_intent is not None:
            intent = followup_intent
            confidence = 1.0

        if (
            self._detect_offer_reply(text) == "decline"
            or intent in {"ad_decline", "objection_no_need"}
        ):
            context["mode"] = "free_chat"
            context["offer_pending"] = False
            context["ad_flow_active"] = False
            context["user_declined_offer_recently"] = True
            context["bridge_pending"] = False
            context["offer_cooldown"] = 0
            context["coffee_chat_turns"] = 0

            reply = self.text_repository.get("product_flow.exit_to_free_chat")
            if not reply:
                reply = "Хорошо, не будем сейчас обсуждать кофемашины. Можем просто пообщаться дальше."

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

        reply = self._execute_product_action(intent=intent, context=context)

        context["last_intent"] = intent
        context["last_confidence"] = confidence

        return {
            "reply": reply,
            "intent": intent,
            "confidence": confidence,
            "context": context,
        }

    def _execute_product_action(
        self,
        intent: str,
        context: dict[str, Any],
    ) -> str:
        """
        Выполняет товарную логику строго через routes/products/response_builder.
        """
        resolved = self.route_engine.resolve(intent)
        route = resolved["route"]
        scenario_template = resolved["scenario_template"]
        global_rules = resolved["global_rules"]

        action = route.get("action")
        scenario = route.get("scenario")

        try:
            if action == "reply":
                self._apply_context_updates(route, context, intent)
                return self.response_builder.build_intent_response(intent)

            if action == "ask_followup":
                self._apply_context_updates(route, context, intent)
                question_key = (
                    scenario_template.get("question_key")
                    or route.get("followup_type")
                )
                return self.response_builder.build_followup_question(question_key)

            if action == "recommend_product":
                selector = route.get("product_selector")
                product = self.recommender.select_product(selector)

                if route.get("save_as_current_product"):
                    context["current_product_id"] = product["id"]

                self._apply_context_updates(route, context, intent)

                ad_text_key = scenario_template.get("ad_text_key", "medium")
                return self.response_builder.build_product_recommendation(
                    product=product,
                    ad_text_key=ad_text_key,
                )

            if action == "compare_products":
                default_products = route.get("default_products")
                products = self.recommender.select_products_for_comparison(
                    default_products=default_products
                )

                self._apply_context_updates(route, context, intent)
                return self.response_builder.build_comparison(products)

            if action == "recommend_alternative":
                current_product_id = context.get("current_product_id")

                if not current_product_id:
                    text = self.text_repository.get("product_flow.no_current_product")
                    if not text:
                        text = "Сначала давай выберем модель, а потом я смогу предложить альтернативу."
                    return text

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

                ad_text_key = scenario_template.get("ad_text_key", "objection_expensive")
                return self.response_builder.build_alternative_recommendation(
                    product=alternative_product,
                    ad_text_key=ad_text_key,
                )

            if action == "describe_current_product":
                current_product_id = context.get("current_product_id")

                if not current_product_id:
                    fallback_action = route.get("fallback_action")
                    if fallback_action == "ask_followup":
                        question_key = route.get("followup_type", "which_product")
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
                    self.response_builder.build_followup_question("budget_or_usage")
                )

            if action == "reply_or_compare":
                self._apply_context_updates(route, context, intent)

                if context.get("current_product_id"):
                    current_product = self.recommender.get_product_by_id(
                        context["current_product_id"]
                    )
                    return (
                        f"Если сомневаешься, могу кратко сравнить варианты. "
                        f"Пока основной вариант — {current_product['name']}."
                    )

                return self.response_builder.build_intent_response(intent)

            if action == "finish_product_flow":
                self._apply_context_updates(route, context, intent)

                context["mode"] = "free_chat"
                context["offer_pending"] = False
                context["ad_flow_active"] = False
                context["bridge_pending"] = False
                context["user_declined_offer_recently"] = False
                context["offer_cooldown"] = 0
                context["coffee_chat_turns"] = 0

                return self.text_repository.get("product_flow.choice_done") or (
                    self.response_builder.build_intent_response(intent)
                )

            return self.text_repository.get("product_flow.fallback") or (
                self.response_builder.build_fallback_response()
            )

        except ProductNotFoundError:
            return self.text_repository.get("product_flow.fallback") or (
                self.response_builder.build_fallback_response()
            )

    def _should_trigger_bridge(
            self,
            text: str,
            context: dict[str, Any],
    ) -> bool:
        """
        Решает, нужно ли сейчас дать реплику-мостик внутри coffee_chat.
        """
        if context.get("user_declined_offer_recently"):
            return False

        if context.get("offer_pending"):
            return False

        if context.get("ad_flow_active"):
            return False

        if context.get("bridge_pending"):
            return False

        if context.get("mode") != "coffee_chat":
            return False

        if context.get("coffee_chat_turns", 0) < 2:
            return False

        return True

    def _should_make_offer(
            self,
            context: dict[str, Any],
    ) -> bool:
        if context.get("user_declined_offer_recently"):
            return False

        if context.get("offer_pending"):
            return False

        if context.get("ad_flow_active"):
            return False

        if not context.get("bridge_pending"):
            return False

        if context.get("mode") != "coffee_chat":
            return False

        if context.get("coffee_chat_turns", 0) < 3:
            return False

        return True

    def _is_explicit_product_entry(
        self,
        text: str,
        intent: str,
        confidence: float,
    ) -> bool:
        """
        Разрешает вход в product_flow только при ЯВНОМ запросе пользователя
        про подбор кофемашины.
        """
        if confidence < self.confidence_threshold:
            return False

        allowed_entry_intents = {
            "ask_recommendation",
            "ask_budget_option",
            "ask_midrange_option",
            "ask_premium_option",
            "ask_home_option",
            "ask_office_option",
            "ask_automatic_option",
            "ask_cappuccino_option",
            "ask_easy_clean_option",
            "ask_compact_option",
            "ask_best_model",
            "ask_comparison",
        }

        if intent not in allowed_entry_intents:
            return False

        return self._contains_product_keywords(text)

    @staticmethod
    def _contains_product_keywords(text: str) -> bool:
        processed = preprocess_text(text)

        keywords = {
            "кофемашина",
            "модель",
            "подобрать",
            "посоветовать",
            "выбрать",
            "для дом",
            "для офис"
        }

        return any(keyword in processed for keyword in keywords)

    @staticmethod
    def _detect_offer_reply(text: str) -> str | None:
        normalized = text.strip().lower()

        accept_phrases = {
            "да",
            "давай",
            "хорошо",
            "ок",
            "ладно",
            "интересно",
            "подбери",
            "посоветуй",
            "да хочу",
            "да помоги",
            "да можно",
        }

        decline_phrases = {
            "нет",
            "не надо",
            "не хочу",
            "не сейчас",
            "неинтересно",
            "не нужно",
            "потом",
            "не будем",
            "неа",
        }

        if normalized in accept_phrases:
            return "accept"

        if normalized in decline_phrases:
            return "decline"

        return None

    @staticmethod
    def _detect_followup_preference_intent(text: str) -> str | None:
        normalized = preprocess_text(text)

        automatic_keywords = {
            "автоматичность",
            "автомат",
            "по один кнопка",
            "один кнопка",
            "по кнопка",
            "все само",
            "всё само",
            "сам готовить",
            "само готовить",
            "минимум ручной действие",
            "нажать кнопка",
            "автоматический",
            "удобство",
            "простота"
        }

        budget_keywords = {
            "цена",
            "бюджет",
            "подешевле",
            "недорого",
            "дешево",
            "дёшево",
            "дешевый",
            "дешёвый",
        }

        cappuccino_keywords = {
            "капучино",
            "латте",
            "пенка",
            "молочный напиток",
            "молочный",
        }

        if any(keyword in normalized for keyword in automatic_keywords):
            return "ask_automatic_option"

        if any(keyword in normalized for keyword in budget_keywords):
            return "ask_budget_option"

        if any(keyword in normalized for keyword in cappuccino_keywords):
            return "ask_cappuccino_option"

        return None

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

    @staticmethod
    def _init_context(context: dict[str, Any]) -> None:
        context.setdefault("mode", "free_chat")
        context.setdefault("offer_pending", False)
        context.setdefault("ad_flow_active", False)
        context.setdefault("bridge_pending", False)
        context.setdefault("user_declined_offer_recently", False)
        context.setdefault("free_chat_turns", 0)
        context.setdefault("offers_shown_count", 0)
        context.setdefault("offer_cooldown", 0)
        context.setdefault("last_offer_topic", None)
        context.setdefault("current_product_id", None)
        context.setdefault("last_intent", None)
        context.setdefault("last_confidence", 0.0)
        context.setdefault("coffee_probe_pending", False)
        context.setdefault("coffee_cooldown", 0)
        context.setdefault("coffee_topic_turns", 0)
        context.setdefault("last_coffee_probe_turn", 0)
        context.setdefault("coffee_chat_turns", 0)

    @staticmethod
    def _detect_coffee_interest_reply(text: str) -> str | None:
        normalized = preprocess_text(text)

        explicit_positive_keywords = {
            "кофе",
            "любить кофе",
            "капучино",
            "латте",
            "эспрессо",
            "кофеман",
            "пить кофе",
            "без кофе",
            "любить капучино",
            "любить латте",
        }

        explicit_negative_keywords = {
            "не любить кофе",
            "не пить кофе",
            "чай",
            "больше чай",
            "кофе не нравиться",
            "не особо кофе",
            "не очень кофе",
        }

        short_positive_answers = {
            "да",
            "ага",
            "угу",
            "конечно",
            "ещё бы",
            "давай",
            "люблю",
            "нравится",
            "нравиться",
            "очень",
            "да нравится",
            "вообще да",
            "в целом да",
            "скорее да",
        }

        short_negative_answers = {
            "нет",
            "неа",
            "не очень",
            "не особо",
            "скорее нет",
            "нет не люблю",
            "не люблю",
            "чай",
            "чай ближе",
        }

        if normalized in short_negative_answers:
            return "negative"

        if normalized in short_positive_answers:
            return "positive"

        if any(keyword in normalized for keyword in explicit_negative_keywords):
            return "negative"

        if any(keyword in normalized for keyword in explicit_positive_keywords):
            return "positive"

        return None

    def _should_make_coffee_probe(
            self,
            context: dict[str, Any],
    ) -> bool:
        if context.get("mode") != "free_chat":
            return False

        if context.get("offer_pending"):
            return False

        if context.get("bridge_pending"):
            return False

        if context.get("ad_flow_active"):
            return False

        if context.get("coffee_probe_pending"):
            return False

        if context.get("coffee_cooldown", 0) > 0:
            return False

        if context.get("free_chat_turns", 0) < 10:
            return False

        if context.get("free_chat_turns", 0) - context.get("last_coffee_probe_turn", 0) < 15:
            return False

        return True

    @staticmethod
    def _tick_cooldowns(context: dict[str, Any]) -> None:
        if context.get("coffee_cooldown", 0) > 0:
            context["coffee_cooldown"] -= 1

    @staticmethod
    def _ensure_sentence_ending(text: str) -> str:
        text = text.strip()
        if not text:
            return text

        if text[-1] not in ".!?":
            text += "."
        return text