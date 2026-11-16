"""Sommelier pipeline: dish -> aroma profile -> menu description."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from olfacto.agent.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    fallback_response,
)


@dataclass
class SommelierResult:
    dish: str
    category: str
    ingredients: list[str]
    missing: list[str]
    aromas: list[dict]
    context: list[dict]
    description: str
    sensory_profile: list[str]
    pairings: list[str]
    confidence: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def make_inference_llm(client, model: str, max_tokens: int = 500) -> Callable:
    """Adapt an InferenceClient into a callable used by the pipeline."""

    def call(messages: list[dict]) -> str:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or ""
            except Exception as error:
                last_error = error
                time.sleep(2**attempt)
        raise last_error if last_error else RuntimeError("LLM call failed")

    return call


def parse_json_response(text: str) -> dict | None:
    """Parse a JSON object, tolerating Markdown code fences."""
    if not text or not text.strip():
        return None

    candidate = text.strip()
    if "```" in candidate:
        blocks = candidate.split("```")
        for block in blocks:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                candidate = block
                break

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


class SommelierAgent:
    """Generate a menu description grounded on odor predictions."""

    def __init__(
        self,
        tools,
        retriever,
        llm: Callable[[list[dict]], str],
        max_retries: int = 1,
    ) -> None:
        self.tools = tools
        self.retriever = retriever
        self.llm = llm
        self.max_retries = max_retries

    def generate(
        self,
        dish: str,
        extra_ingredients: list[str] | None = None,
        top_k: int = 8,
    ) -> SommelierResult:
        profile = self.tools.profile(dish, extra_ingredients, top_k=top_k)
        aromas = profile["aromas"]
        if not aromas:
            raise ValueError("no odor predictions for this dish")

        ingredient_names = [row["name"] for row in profile["ingredients"]]
        query = " ".join([dish, *ingredient_names])
        context = self.retriever.retrieve(query, k=3)
        pairings = context[0]["pairings"] if context else []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    dish, profile["category"], ingredient_names, aromas, context
                ),
            },
        ]

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm(messages)
            except Exception:
                break

            data = parse_json_response(raw)
            if data is not None and self._is_valid(data, aromas):
                return SommelierResult(
                    dish=dish,
                    category=profile["category"],
                    ingredients=ingredient_names,
                    missing=profile["missing"],
                    aromas=aromas,
                    context=context,
                    description=data["descricao"],
                    sensory_profile=data["perfil_sensorial"],
                    pairings=data["harmonizacoes"],
                    confidence=data["confianca"],
                    source="llm",
                )

            if attempt < self.max_retries:
                messages = messages[:2] + [
                    {
                        "role": "user",
                        "content": (
                            "O JSON anterior é inválido ou cita aromas fora da lista. "
                            "Responda novamente usando SOMENTE os aromas previstos."
                        ),
                    }
                ]

        fallback = fallback_response(dish, aromas, pairings)
        return SommelierResult(
            dish=dish,
            category=profile["category"],
            ingredients=ingredient_names,
            missing=profile["missing"],
            aromas=aromas,
            context=context,
            description=fallback["descricao"],
            sensory_profile=fallback["perfil_sensorial"],
            pairings=fallback["harmonizacoes"],
            confidence=fallback["confianca"],
            source="fallback",
        )

    def _is_valid(self, data: dict, aromas: list[dict]) -> bool:
        allowed = {item["label_pt_br"].lower() for item in aromas}
        profile = data.get("perfil_sensorial")
        if not isinstance(profile, list) or not 3 <= len(profile) <= 5:
            return False
        if any(
            not isinstance(item, str) or item.lower() not in allowed
            for item in profile
        ):
            return False

        pairings = data.get("harmonizacoes")
        if (
            not isinstance(pairings, list)
            or len(pairings) != 3
            or any(not isinstance(item, str) or not item.strip() for item in pairings)
        ):
            return False

        if data.get("confianca") not in {"alta", "media", "baixa"}:
            return False

        description = data.get("descricao")
        return isinstance(description, str) and 20 <= len(description) <= 600
