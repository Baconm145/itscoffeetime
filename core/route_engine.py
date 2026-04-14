from typing import Any


class RouteEngineError(Exception):
    """Базовая ошибка route engine."""


class RouteNotFoundError(RouteEngineError):
    """Ошибка, если маршрут не найден."""


class RouteEngine:
    """
    Работает с dialogue_routes.json:
    - получает маршрут по интенту
    - отдаёт сценарии, вопросы и глобальные правила
    """

    def __init__(self, routes_data: dict[str, Any]) -> None:
        self.routes_data = routes_data
        self.routes = routes_data.get("routes", {})
        self.scenario_templates = routes_data.get("scenario_templates", {})
        self.followup_questions = routes_data.get("followup_questions", {})
        self.global_rules = routes_data.get("global_rules", {})

    def get_route(self, intent: str) -> dict[str, Any]:
        """
        Возвращает маршрут по интенту.
        Если интент не найден, возвращает fallback route.
        """
        if intent in self.routes:
            return self.routes[intent]

        if "fallback" in self.routes:
            return self.routes["fallback"]

        raise RouteNotFoundError(
            f"Маршрут для интента '{intent}' не найден, и fallback отсутствует."
        )

    def get_scenario_template(self, scenario_name: str) -> dict[str, Any]:
        """
        Возвращает шаблон сценария по имени.
        """
        return self.scenario_templates.get(scenario_name, {})

    def get_followup_questions(self, question_key: str) -> list[str]:
        """
        Возвращает список уточняющих вопросов по ключу.
        """
        questions = self.followup_questions.get(question_key, [])
        if isinstance(questions, list):
            return questions
        return []

    def get_global_rules(self) -> dict[str, Any]:
        """
        Возвращает глобальные правила маршрутизации.
        """
        return self.global_rules

    def resolve(self, intent: str) -> dict[str, Any]:
        """
        Полностью собирает информацию по интенту:
        - route
        - scenario_template
        """
        route = self.get_route(intent)
        scenario_name = route.get("scenario")
        scenario_template = self.get_scenario_template(scenario_name) if scenario_name else {}

        return {
            "intent": intent,
            "route": route,
            "scenario_template": scenario_template,
            "global_rules": self.get_global_rules(),
        }