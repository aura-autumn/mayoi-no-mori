"""
heatmap.py

Two visualizations:
1. Sentence heatmap — color-codes the primary answer by confabulation risk
2. Drift line chart — shows confidence score per sentence across the answer

These are the "LinkedIn visuals" — designed to tell the story at a glance.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Color palette: green = solid, yellow = uncertain, red = confabulated
COLORS = {
    "SOLID":        "#2ECC71",
    "UNCERTAIN":    "#F39C12",
    "CONFABULATED": "#E74C3C",
}

CONFIDENCE_COLOR = "#3498DB"


def plot_sentence_heatmap(
    sentences: list[str],
    confidence: list[float],
    labels: list[str],
    question: str,
    overall_drift: float,
    save_path: str = None,
) -> plt.Figure:
    """
    Renders a visual where each sentence of the answer is shown as a
    color-coded block based on confabulation risk.
    """
    fig, ax = plt.subplots(figsize=(14, max(4, len(sentences) * 1.1)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(sentences))
    ax.axis("off")

    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    for i, (sent, label) in enumerate(zip(sentences, labels)):
        y = len(sentences) - i - 1
        color = COLORS[label]

        # Background bar
        bar = mpatches.FancyBboxPatch(
            (0.01, y + 0.05), 0.98, 0.85,
            boxstyle="round,pad=0.01",
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(bar)

        # Sentence text
        wrapped = _wrap_text(sent, max_chars=110)
        ax.text(
            0.03, y + 0.48, wrapped,
            va="center", ha="left",
            fontsize=9, color="white",
            fontfamily="monospace",
        )

        # Risk label on right
        ax.text(
            0.97, y + 0.48, label,
            va="center", ha="right",
            fontsize=8, color=color,
            fontweight="bold",
        )

    # Title
    short_q = question if len(question) < 80 else question[:77] + "..."
    fig.suptitle(
        f'Confabulation Analysis\n"{short_q}"\nOverall Drift Risk: {overall_drift:.1%} — {_drift_label(overall_drift)}',
        color="white", fontsize=11, y=0.98, va="top"
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["SOLID"],        label="SOLID — consistent across runs"),
        mpatches.Patch(color=COLORS["UNCERTAIN"],    label="UNCERTAIN — some drift detected"),
        mpatches.Patch(color=COLORS["CONFABULATED"], label="CONFABULATED — high drift, likely made up"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=3,
              facecolor="#1A1A2E", edgecolor="white",
              labelcolor="white", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_confidence_curve(
    sentences: list[str],
    confidence: list[float],
    question: str,
    save_path: str = None,
) -> plt.Figure:
    """
    Line chart showing confidence score per sentence.
    The spike downward = the moment the model starts confabulating.
    This is the key "LinkedIn graph" that tells the whole story in one glance.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    x = np.arange(len(confidence))
    ax.plot(x, confidence, color=CONFIDENCE_COLOR, linewidth=2.5, marker="o", markersize=6, zorder=3)

    # Fill zones
    ax.fill_between(x, confidence, 0.80, where=[c >= 0.80 for c in confidence],
                    alpha=0.3, color=COLORS["SOLID"], label="SOLID (≥0.80)")
    ax.fill_between(x, confidence, 0.60, where=[0.60 <= c < 0.80 for c in confidence],
                    alpha=0.3, color=COLORS["UNCERTAIN"], label="UNCERTAIN (0.60–0.80)")
    ax.fill_between(x, confidence, 0, where=[c < 0.60 for c in confidence],
                    alpha=0.3, color=COLORS["CONFABULATED"], label="CONFABULATED (<0.60)")

    # Threshold lines
    ax.axhline(0.80, color=COLORS["SOLID"],        linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.60, color=COLORS["CONFABULATED"], linestyle="--", linewidth=1, alpha=0.6)

    # X-axis labels: short version of each sentence
    short_labels = [f"S{i+1}" for i in range(len(sentences))]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, color="white", fontsize=8)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], color="white", fontsize=8)
    ax.set_ylim(0, 1.05)

    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.tick_params(colors="white")
    ax.set_xlabel("Sentence", color="white", fontsize=10)
    ax.set_ylabel("Consistency Score", color="white", fontsize=10)

    short_q = question if len(question) < 70 else question[:67] + "..."
    ax.set_title(f'Sentence-Level Confidence\n"{short_q}"', color="white", fontsize=11)

    ax.legend(facecolor="#1A1A2E", edgecolor="white", labelcolor="white", fontsize=8)
    ax.grid(axis="y", color="#333", linewidth=0.5, zorder=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _wrap_text(text: str, max_chars: int = 110) -> str:
    """Wrap long sentences for display."""
    if len(text) <= max_chars:
        return text
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            lines.append(current.strip())
            current = word
        else:
            current += " " + word
    if current:
        lines.append(current.strip())
    return "\n".join(lines)


def _drift_label(drift: float) -> str:
    if drift < 0.15:
        return "🟢 LOW RISK"
    elif drift < 0.35:
        return "🟡 MEDIUM RISK"
    else:
        return "🔴 HIGH RISK"