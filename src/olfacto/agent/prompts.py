"""Prompt templates and fallback responses for the Sommelier."""

from __future__ import annotations

SYSTEM_PROMPT = """Você é o Olfacto Sommelier, um assistente de descrições sensoriais para menus.

Regras:
- Use apenas os aromas previstos na lista fornecida. Não invente aromas.
- Não faça alegações de saúde, alergênicos ou segurança alimentar.
- Se a confiança for baixa, diga isso de forma simples.
- Escreva em português do Brasil, com no máximo 80 palavras na descrição.
- Use linguagem sensorial e apetitosa, com textura e aroma, sem clichês.
- Nas harmonizações, varie as categorias: acompanhamento, bebida e finalização.
- Não repita bebidas alcoólicas nas harmonizações.
- Responda somente com JSON válido, sem texto fora do JSON.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "descricao": {"type": "string"},
        "perfil_sensorial": {"type": "array", "items": {"type": "string"}},
        "harmonizacoes": {"type": "array", "items": {"type": "string"}},
        "confianca": {"type": "string", "enum": ["alta", "media", "baixa"]},
    },
    "required": ["descricao", "perfil_sensorial", "harmonizacoes", "confianca"],
}


def build_user_prompt(
    dish: str,
    category: str,
    ingredients: list[str],
    aromas: list[dict],
    contexts: list[dict],
) -> str:
    aroma_lines = "\n".join(
        f"- {item['label_pt_br']} ({item['probability']:.0%})" for item in aromas
    )
    context_lines = "\n".join(
        f"- {item['dish']}: {item['description']}" for item in contexts
    )
    if not context_lines:
        context_lines = "- Nenhum contexto recuperado."

    return f"""Prato: {dish}
Categoria: {category}
Ingredientes: {", ".join(ingredients)}

Aromas previstos pelo modelo (use SOMENTE estes):
{aroma_lines}

Contexto sensorial recuperado:
{context_lines}

Gere:
1. descricao: 1 a 2 frases para menu, no máximo 80 palavras.
2. perfil_sensorial: 3 a 5 aromas da lista acima, em português.
3. harmonizacoes: 3 sugestões variadas — um acompanhamento, uma bebida e
   uma finalização. Não repita a mesma categoria.
4. confianca: alta, media ou baixa.

Formato de saída (JSON):
{{
  "descricao": "...",
  "perfil_sensorial": ["...", "..."],
  "harmonizacoes": ["...", "...", "..."],
  "confianca": "alta"
}}
"""


def fallback_response(dish: str, aromas: list[dict], pairings: list[str]) -> dict:
    names = [item["label_pt_br"] for item in aromas[:3]]
    if not names:
        names = ["neutro"]

    if len(names) == 1:
        description = f"{dish} com aroma de {names[0]}, em uma proposta simples."
    else:
        description = (
            f"{dish} com aroma de {names[0]}, notas de "
            f"{', '.join(names[1:])} e final equilibrado."
        )

    return {
        "descricao": description,
        "perfil_sensorial": names,
        "harmonizacoes": pairings[:3] or ["arroz branco", "salada", "água"],
        "confianca": "baixa",
    }
