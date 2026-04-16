import random
from typing import Any

from config import DEFAULT_AD_TEXT_KEY


class ResponseBuilder:
    """
    Собирает текстовые ответы бота из dialogue_routes.json и products.json.
    """

    def __init__(self, routes_data: dict[str, Any]) -> None:
        self.routes_data = routes_data
        self.followup_questions = self.routes_data.get("followup_questions", {})

    def build_followup_question(self, question_key: str) -> str:
        """
        Возвращает случайный уточняющий вопрос по ключу.
        """
        questions = self.followup_questions.get(question_key, [])
        if not questions:
            return "Уточни, пожалуйста, что для тебя важнее."

        return random.choice(questions)

    def build_product_recommendation(
        self,
        product: dict[str, Any],
        ad_text_key: str = DEFAULT_AD_TEXT_KEY,
    ) -> str:
        """
        Собирает рекомендацию товара на основе ad_texts.
        """
        name = product.get("name", "Неизвестная модель")
        brand = product.get("brand", "Неизвестный бренд")

        ad_texts = product.get("ad_texts", {})
        main_text = ad_texts.get(ad_text_key)

        if not main_text:
            main_text = f"Могу предложить {brand} {name} как подходящий вариант."

        return main_text

    def build_alternative_recommendation(
        self,
        product: dict[str, Any],
        ad_text_key: str = "objection_expensive",
    ) -> str:
        """
        Собирает ответ с альтернативным товаром.
        """
        ad_texts = product.get("ad_texts", {})
        main_text = ad_texts.get(ad_text_key)

        if not main_text:
            main_text = (
                f"Можно рассмотреть альтернативу: "
                f"{product.get('name', 'неизвестную модель')}."
            )

        return main_text

    def build_product_features(self, product: dict[str, Any]) -> str:
        """
        Собирает описание характеристик товара.
        """
        name = product.get("name", "Неизвестная модель")
        features = product.get("features", [])
        advantages = product.get("advantages", [])

        parts: list[str] = [f"{name}:"]

        if features:
            feature_text = ", ".join(features[:5])
            parts.append(f"Основные функции: {feature_text}.")

        if advantages:
            advantages_text = ", ".join(advantages[:3])
            parts.append(f"Преимущества: {advantages_text}.")

        return " ".join(parts)

    def build_product_price_description(self, product: dict[str, Any]) -> str:
        """
        Собирает описание ценового сегмента товара.
        """
        name = product.get("name", "Неизвестная модель")
        budget_tier = product.get("budget_tier", "unknown")

        tier_map = {
            "budget": "бюджетному",
            "mid": "среднему",
            "premium": "премиальному",
        }

        tier_text = tier_map.get(budget_tier, "неизвестному")

        return f"{name} относится к {tier_text} ценовому сегменту."

    def build_comparison(self, products: list[dict[str, Any]]) -> str:
        """
        Собирает краткое сравнение нескольких товаров.
        """
        if not products:
            return "Сейчас не удалось подобрать товары для сравнения."

        lines = ["Вот краткое сравнение моделей:"]

        for product in products:
            name = product.get("name", "Неизвестная модель")
            budget_tier = product.get("budget_tier", "unknown")
            automation_level = product.get("automation_level", "unknown")
            milk_system = product.get("milk_system", "unknown")

            lines.append(
                f"- {name}: сегмент = {budget_tier}, "
                f"уровень автоматизации = {automation_level}, "
                f"молочная система = {milk_system}."
            )

        return "\n".join(lines)

    @staticmethod
    def build_fallback_response() -> str:
        """
        Возвращает универсальный fallback-ответ.
        """
        return (
            "Не до конца понял запрос. "
            "Могу помочь выбрать кофемашину, сравнить модели или подобрать вариант по бюджету."
        )
