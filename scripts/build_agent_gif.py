"""Build a short GIF from Olfacto Sommelier outputs."""

from __future__ import annotations

import textwrap
from pathlib import Path

from huggingface_hub import InferenceClient, get_token
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from olfacto.agent.pipeline import SommelierAgent, make_inference_llm
from olfacto.agent.retrieval import SensoryRetriever, load_corpus
from olfacto.agent.tools import OdorTools
from olfacto.ensemble import EnsemblePredictor

DISHES = ["Feijoada", "Pizza margherita", "Café"]
SIZE = (1200, 700)
GREEN = "#4c9f70"
AMBER = "#e6b566"
INK = "#1f2937"
MUTED = "#6b7280"
BAR_COLORS = ["#4c9f70", "#7fb069", "#e6b566", "#d99a4e", "#c1440e"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    properties = font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal"
    )
    return ImageFont.truetype(font_manager.findfont(properties), size)


def title_frame() -> Image.Image:
    image = Image.new("RGB", SIZE, "#f7f9f5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, SIZE[0], 14), fill=GREEN)
    draw.text((60, 120), "Olfacto", font=font(86, True), fill=INK)
    draw.text((445, 120), "Sommelier", font=font(86, True), fill=GREEN)
    draw.text(
        (64, 245),
        "Do cheiro ao menu, em português.",
        font=font(36),
        fill="#4b5563",
    )
    draw.text(
        (64, 330),
        "45 pratos · 20 avaliações · 75% de preferência sobre o template",
        font=font(24),
        fill=MUTED,
    )
    draw.text(
        (64, 380),
        "Modelo de odor + RAG + LLM com guardrails",
        font=font(24),
        fill=MUTED,
    )
    draw.text(
        (64, 520),
        "huggingface.co/spaces/SamuelCagnani/olfacto-agent",
        font=font(22, True),
        fill=INK,
    )
    return image


def dish_frame(dish: str, result) -> Image.Image:
    image = Image.new("RGB", SIZE, "#f7f9f5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, SIZE[0], 86), fill=INK)
    draw.text((40, 26), "Olfacto Sommelier", font=font(32, True), fill="white")
    draw.text((40, 112), dish, font=font(38, True), fill=INK)

    badge = "LLM" if result.source == "llm" else "FALLBACK"
    badge_color = GREEN if result.source == "llm" else AMBER
    draw.rounded_rectangle((980, 112, 1150, 154), radius=12, fill=badge_color)
    draw.text((1006, 121), badge, font=font(20, True), fill="white")

    draw.text((40, 185), "DESCRIÇÃO PARA O MENU", font=font(18, True), fill=GREEN)
    description = textwrap.fill(result.description, 58)
    draw.multiline_text((40, 218), description, font=font(22), fill="#374151", spacing=10)

    draw.text((40, 380), "PERFIL SENSORIAL", font=font(18, True), fill=GREEN)
    for index, aroma in enumerate(result.aromas[:5]):
        y = 420 + index * 44
        draw.text((40, y), aroma["label_pt_br"], font=font(19), fill=INK)
        width = int(260 * max(0.05, min(1.0, aroma["probability"])))
        draw.rounded_rectangle(
            (220, y + 6, 220 + width, y + 22),
            radius=8,
            fill=BAR_COLORS[index % len(BAR_COLORS)],
        )
        draw.text(
            (500, y),
            f"{aroma['probability']:.0%}",
            font=font(18, True),
            fill=MUTED,
        )

    draw.text((650, 380), "HARMONIZAÇÕES", font=font(18, True), fill=GREEN)
    for index, pairing in enumerate(result.pairings):
        draw.text(
            (650, 420 + index * 40),
            f"• {pairing}",
            font=font(20),
            fill="#374151",
        )

    draw.text(
        (650, 600),
        "Perfil 100% fiel · descrição gerada a partir dos aromas previstos",
        font=font(16),
        fill=MUTED,
    )
    return image


def main() -> int:
    token = get_token()
    if not token:
        raise SystemExit("no Hugging Face token configured")

    predictor = EnsemblePredictor.from_pretrained(
        "outputs/cv/final",
        "data/odor_labels_ptbr.json",
        "outputs/fingerprint/model.joblib",
        weight_fingerprint=0.8,
    )
    tools = OdorTools(
        predictor,
        ["data/brazilian_ingredients.csv", "data/sommelier_ingredients.csv"],
        "data/dish_ingredients.json",
    )
    retriever = SensoryRetriever(load_corpus("data/sensory_corpus.jsonl"))
    client = InferenceClient(
        model="Qwen/Qwen2.5-7B-Instruct", provider="featherless-ai", token=token
    )
    agent = SommelierAgent(
        tools,
        retriever,
        make_inference_llm(client, "Qwen/Qwen2.5-7B-Instruct"),
    )

    frames = [title_frame()]
    for dish in DISHES:
        result = agent.generate(dish)
        for _ in range(2):
            if result.source == "llm":
                break
            result = agent.generate(dish)
        frames.append(dish_frame(dish, result))
        print(dish, result.source)

    output = Path("docs/images/agent-demo.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=2400,
        loop=0,
        optimize=True,
    )
    print(output, output.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
