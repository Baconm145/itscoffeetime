from typing import Any


class RecommenderError(Exception):
    """Базовая ошибка recommender."""


class ProductNotFoundError(RecommenderError):
    """Ошибка, если товар не найден."""


class Recommender:
    """
    Работает с товарами из products.json:
    - выбирает товар по selector
    - ищет альтернативу
    - возвращает товары для сравнения
    """

    def __init__(self, products_data: dict[str, Any]) -> None:
        self.products_data = products_data
        self.products = products_data.get("products", [])

        self.products_by_id = {
            product["id"]: product
            for product in self.products
            if isinstance(product, dict) and "id" in product
        }

    def get_product_by_id(self, product_id: str) -> dict[str, Any]:
        """
        Возвращает товар по id.
        """
        product = self.products_by_id.get(product_id)
        if not product:
            raise ProductNotFoundError(f"Товар с id '{product_id}' не найден.")
        return product

    def select_product(self, selector: dict[str, Any] | None) -> dict[str, Any]:
        """
        Выбирает один товар по selector.
        Поддерживает:
        - preferred_order
        - budget_tier
        - use_cases
        - drink_focus
        - scenario_tags
        """
        if not selector:
            raise ProductNotFoundError("Пустой product selector.")

        # 1. Если задан preferred_order — пробуем сначала его
        preferred_order = selector.get("preferred_order")
        if isinstance(preferred_order, list):
            for product_id in preferred_order:
                if product_id in self.products_by_id:
                    product = self.products_by_id[product_id]
                    if self._matches_selector(product, selector, ignore_keys={"preferred_order"}):
                        return product

        # 2. Иначе просто ищем первый подходящий товар
        for product in self.products:
            if self._matches_selector(product, selector):
                return product

        raise ProductNotFoundError(f"Не найден товар по selector: {selector}")

    def select_products_for_comparison(
        self,
        product_ids: list[str] | None = None,
        default_products: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Возвращает список товаров для сравнения.
        Если product_ids не переданы, использует default_products.
        """
        result: list[dict[str, Any]] = []

        ids_to_use = product_ids or default_products or []

        for product_id in ids_to_use:
            if product_id in self.products_by_id:
                result.append(self.products_by_id[product_id])

        if not result:
            raise ProductNotFoundError("Не удалось подобрать товары для сравнения.")

        return result

    def find_alternative(
        self,
        current_product_id: str,
        alternative_rule: str,
        global_rules: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Находит альтернативный товар на основе global_rules и budget tier текущего товара.

        Поддерживаемые alternative_rule:
        - lower_budget
        - adjacent_budget
        """
        current_product = self.get_product_by_id(current_product_id)
        current_budget_tier = current_product.get("budget_tier")

        alternative_rules = global_rules.get("alternative_rules", {})
        rule_mapping = alternative_rules.get(alternative_rule)

        if not isinstance(rule_mapping, dict):
            raise ProductNotFoundError(
                f"Правило альтернативы '{alternative_rule}' не найдено."
            )

        target_budget_tier = rule_mapping.get(current_budget_tier)
        if not target_budget_tier:
            raise ProductNotFoundError(
                f"Не найден target budget tier для '{current_budget_tier}'."
            )

        for product in self.products:
            if product.get("budget_tier") == target_budget_tier:
                return product

        raise ProductNotFoundError(
            f"Альтернативный товар для budget tier '{target_budget_tier}' не найден."
        )

    def _matches_selector(
        self,
        product: dict[str, Any],
        selector: dict[str, Any],
        ignore_keys: set[str] | None = None,
    ) -> bool:
        """
        Проверяет, подходит ли товар под selector.
        """
        ignore_keys = ignore_keys or set()

        for key, value in selector.items():
            if key in ignore_keys:
                continue

            product_value = product.get(key)

            if isinstance(product_value, list):
                if value not in product_value:
                    return False
            else:
                if product_value != value:
                    return False

        return True