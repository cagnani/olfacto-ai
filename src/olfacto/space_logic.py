"""Pure helpers shared by the Gradio app and its tests."""

from __future__ import annotations

import math

import pandas as pd

FAMILY_COLORS = {
    "floral": "#d66ba0",
    "frutado": "#f2a541",
    "verde": "#4c9f70",
    "amadeirado": "#8b5e3c",
    "especiado": "#c1440e",
    "doce": "#e6b566",
    "animal": "#6b4c3b",
    "químico": "#5b7db1",
    "terroso": "#7a6a53",
    "alimentar": "#b5651d",
    "outro": "#9aa0a6",
}


def predictions_table(predictions: list) -> list[dict]:
    """Convert predictions into rows for a Gradio dataframe."""
    return [
        {
            "aroma": prediction.label_pt_br,
            "probabilidade": round(prediction.probability, 3),
            "rótulo": prediction.label,
        }
        for prediction in predictions
    ]


def odor_wheel_svg(
    predictions: list,
    lexicon: dict[str, dict[str, str]],
    size: int = 360,
) -> str:
    """Render a simple radial bar chart as inline SVG."""
    if not predictions:
        return "<p>Sem previsões.</p>"

    center = size / 2
    max_radius = size * 0.38
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f'<circle cx="{center}" cy="{center}" r="{max_radius}" '
        'fill="none" stroke="#e5e7eb" stroke-width="1"/>',
    ]

    count = len(predictions)
    for index, prediction in enumerate(predictions):
        angle = -math.pi / 2 + 2 * math.pi * index / count
        radius = max(0.03, min(1.0, prediction.probability)) * max_radius
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        family = lexicon[prediction.label]["family"]
        color = FAMILY_COLORS.get(family, FAMILY_COLORS["outro"])

        parts.append(
            f'<line x1="{center:.1f}" y1="{center:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="9" stroke-linecap="round"/>'
        )
        label_x = center + (max_radius + 22) * math.cos(angle)
        label_y = center + (max_radius + 22) * math.sin(angle)
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-size="11" '
            f'text-anchor="middle" fill="#374151">{prediction.label_pt_br}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def find_ingredient(catalog: pd.DataFrame, name: str) -> dict | None:
    """Return the catalog row for an ingredient name, or None."""
    match = catalog[catalog["name_pt"] == name]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def scatter_svg(
    points: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    family_col: str = "family",
    label_col: str = "name",
    size: int = 560,
) -> str:
    """Render a static 2D scatter plot as inline SVG."""
    if points.empty:
        return "<p>Mapa indisponível.</p>"

    x = points[x_col].astype(float)
    y = points[y_col].astype(float)
    x_span = (x.max() - x.min()) or 1.0
    y_span = (y.max() - y.min()) or 1.0
    pad = 30

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#fafafa" rx="12"/>',
    ]
    for _, row in points.iterrows():
        px = pad + (float(row[x_col]) - x.min()) / x_span * (size - 2 * pad)
        py = size - pad - (float(row[y_col]) - y.min()) / y_span * (size - 2 * pad)
        color = FAMILY_COLORS.get(row[family_col], FAMILY_COLORS["outro"])
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}" '
            f'fill-opacity="0.75"><title>{row[label_col]}</title></circle>'
        )
    parts.append("</svg>")
    return "".join(parts)
