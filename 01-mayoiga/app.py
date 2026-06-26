"""
app.py — Mayoiga UI

Full mystical Japanese folklore aesthetic.
Fog layers, torii gate, ancient typography, animated mist.
Runs as a Gradio app at localhost:7860.
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from core.detector import collect_answers
from core.scorer import score_drift, classify_sentences
from examples.demo_questions import GROUNDED_QUESTIONS, CONFABULATION_TRAPS

# Interleave grounded and trap questions so the UI shows the contrast clearly
EXAMPLE_QUESTIONS = GROUNDED_QUESTIONS[:3] + CONFABULATION_TRAPS[:3]

# ── Color palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "bg":           "#0A0A14",
    "mist":         "#12122A",
    "panel":        "#0F0F20",
    "border":       "#2A2A4A",
    "gold":         "#C9A84C",
    "gold_dim":     "#7A6030",
    "ghost":        "#B8B8CC",
    "solid":        "#4A7C59",
    "solid_bright": "#6BAA7A",
    "uncertain":    "#8B6914",
    "uncertain_bright": "#C9A030",
    "confab":       "#6B1A1A",
    "confab_bright": "#C04040",
    "text":         "#D0D0E8",
    "text_dim":     "#707090",
}


def run_detection(question: str, n_variants: int):
    if not question.strip():
        return None, None, _empty_summary()

    result = collect_answers(question, n_variants=int(n_variants))
    scores = score_drift(result["answers"])

    if not scores["sentences"]:
        return None, None, _error_summary()

    labels = classify_sentences(scores["sentences"], scores["confidence"])
    heatmap_fig = _plot_heatmap(scores, labels, question)
    curve_fig = _plot_curve(scores, question)
    summary = _build_summary(question, scores, labels, result["variants"])

    return heatmap_fig, curve_fig, summary


def _plot_heatmap(scores, labels, question):
    sentences = scores["sentences"]
    confidence = scores["confidence"]
    n = len(sentences)

    fig, ax = plt.subplots(figsize=(14, max(5, n * 1.2)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")

    color_map = {
        "SOLID":        (PALETTE["solid"],        PALETTE["solid_bright"]),
        "UNCERTAIN":    (PALETTE["uncertain"],     PALETTE["uncertain_bright"]),
        "CONFABULATED": (PALETTE["confab"],        PALETTE["confab_bright"]),
    }
    symbol_map = {
        "SOLID": "✦", "UNCERTAIN": "◈", "CONFABULATED": "✧"
    }

    for i, (sent, label, conf) in enumerate(zip(sentences, labels, confidence)):
        y = n - i - 1
        bg_color, accent = color_map[label]

        # Glow bar
        bar = mpatches.FancyBboxPatch(
            (0.01, y + 0.06), 0.98, 0.84,
            boxstyle="round,pad=0.008",
            facecolor=bg_color,
            alpha=0.20,
            edgecolor=accent,
            linewidth=0.8,
        )
        ax.add_patch(bar)

        # Left accent stripe
        stripe = mpatches.FancyBboxPatch(
            (0.01, y + 0.06), 0.004, 0.84,
            boxstyle="round,pad=0.001",
            facecolor=accent,
            alpha=0.9,
        )
        ax.add_patch(stripe)

        # Symbol
        ax.text(0.025, y + 0.48, symbol_map[label],
                va="center", ha="left", fontsize=11,
                color=accent, alpha=0.9)

        # Sentence text
        wrapped = _wrap(sent, 108)
        ax.text(0.045, y + 0.48, wrapped,
                va="center", ha="left",
                fontsize=8.5, color=PALETTE["text"],
                fontfamily="monospace", linespacing=1.4)

        # Confidence badge
        ax.text(0.97, y + 0.75, f"{conf:.0%}",
                va="center", ha="right",
                fontsize=7.5, color=accent, fontweight="bold")
        ax.text(0.97, y + 0.25, label,
                va="center", ha="right",
                fontsize=7, color=accent, alpha=0.7)

    # Title block
    drift = scores["overall_drift"]
    risk_label = _risk_label(drift)
    short_q = question if len(question) < 75 else question[:72] + "…"

    fig.text(0.5, 0.99,
             "迷い家  MAYOIGA — Confabulation Analysis",
             ha="center", va="top",
             fontsize=13, color=PALETTE["gold"],
             fontweight="bold")
    fig.text(0.5, 0.965,
             f'"{short_q}"',
             ha="center", va="top",
             fontsize=9, color=PALETTE["ghost"], style="italic")
    fig.text(0.5, 0.945,
             f"Drift Risk: {drift:.1%}  ·  {risk_label}",
             ha="center", va="top",
             fontsize=9, color=_risk_color(drift))

    # Legend
    legend_items = [
        mpatches.Patch(color=PALETTE["solid_bright"],   label="✦ SOLID — knowledge holds"),
        mpatches.Patch(color=PALETTE["uncertain_bright"], label="◈ UNCERTAIN — edges blur"),
        mpatches.Patch(color=PALETTE["confab_bright"],  label="✧ CONFABULATED — phantom answer"),
    ]
    ax.legend(handles=legend_items,
              loc="lower center", bbox_to_anchor=(0.5, -0.06),
              ncol=3, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"],
              labelcolor=PALETTE["ghost"], fontsize=8,
              framealpha=0.9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    return fig


def _plot_curve(scores, question):
    sentences = scores["sentences"]
    confidence = scores["confidence"]
    n = len(sentences)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["mist"])

    # Fill zones
    conf_arr = np.array(confidence)
    ax.fill_between(x, conf_arr, 0.80,
                    where=conf_arr >= 0.80,
                    alpha=0.25, color=PALETTE["solid_bright"])
    ax.fill_between(x, conf_arr, 0.60,
                    where=(conf_arr >= 0.60) & (conf_arr < 0.80),
                    alpha=0.25, color=PALETTE["uncertain_bright"])
    ax.fill_between(x, conf_arr, 0,
                    where=conf_arr < 0.60,
                    alpha=0.25, color=PALETTE["confab_bright"])

    # Threshold lines
    ax.axhline(0.80, color=PALETTE["solid_bright"],   linestyle="--", lw=0.8, alpha=0.5)
    ax.axhline(0.60, color=PALETTE["confab_bright"],  linestyle="--", lw=0.8, alpha=0.5)

    # Main line
    ax.plot(x, confidence,
            color=PALETTE["gold"], linewidth=2, marker="o",
            markersize=7, markerfacecolor=PALETTE["gold_dim"],
            markeredgecolor=PALETTE["gold"], zorder=5)

    # Point labels
    for i, (xi, ci) in enumerate(zip(x, confidence)):
        color = _point_color(ci)
        ax.annotate(f"{ci:.0%}",
                    (xi, ci), textcoords="offset points",
                    xytext=(0, 10), ha="center",
                    fontsize=7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(n)],
                       color=PALETTE["ghost"], fontsize=8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"],
                       color=PALETTE["ghost"], fontsize=8)
    ax.set_ylim(0, 1.1)

    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    ax.tick_params(colors=PALETTE["ghost"])
    ax.set_xlabel("Sentence", color=PALETTE["text_dim"], fontsize=9)
    ax.set_ylabel("Consistency Score", color=PALETTE["text_dim"], fontsize=9)
    ax.grid(axis="y", color=PALETTE["border"], linewidth=0.5, alpha=0.5)

    short_q = question if len(question) < 65 else question[:62] + "…"
    ax.set_title(f'狐火  Sentence Consistency · "{short_q}"',
                 color=PALETTE["gold"], fontsize=10, pad=12)

    plt.tight_layout()
    return fig


def _build_summary(question, scores, labels, variants):
    risk = scores["label"]
    drift = scores["overall_drift"]
    emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    sym = {"SOLID": "✦", "UNCERTAIN": "◈", "CONFABULATED": "✧"}

    lines = [
        f"## 迷い家 — Analysis Complete",
        f"",
        f"**Question:** {question}",
        f"",
        f"**Overall Drift Risk: {emoji[risk]} {risk} — {drift:.1%}**",
        f"",
        f"---",
        f"",
        f"### Sentence Reading",
        f"",
    ]

    for sent, label, conf in zip(scores["sentences"], labels, scores["confidence"]):
        lines.append(f"{sym[label]} `{label} {conf:.0%}` — {sent}")
        lines.append("")

    lines += ["---", "", "### What the Forest Says", ""]

    if risk == "LOW":
        lines.append(
            "The model's answers held consistent across all variants. "
            "This ground is solid — the knowledge appears real, not constructed. "
            "The house you found is not a phantom."
        )
    elif risk == "MEDIUM":
        lines.append(
            "Partial drift detected. Some sentences remain stable; others waver. "
            "The model may be mixing genuine recall with confident inference. "
            "Verify the specific details that scored UNCERTAIN before trusting them."
        )
    else:
        lines.append(
            "High drift across variants. The model gave substantially different answers "
            "each time the question was rephrased — a clear signal that it has no ground "
            "truth to anchor to. It built you a phantom house. The specific details "
            "in this answer are likely invented. Do not trust them without verification."
        )

    lines += [
        "", "---", "",
        f"*{len(variants)} question variants tested*",
        "", "**Variants used:**", "",
    ]
    for v in variants:
        lines.append(f"- *{v}*")

    return "\n".join(lines)


def _empty_summary():
    return "### 迷い家\n\n*Enter a question and step into the forest.*"

def _error_summary():
    return "### 迷い家\n\n⚠️ Could not parse response into sentences. Try a longer or more specific question."

def _wrap(text, n):
    if len(text) <= n:
        return text
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur.strip())
            cur = w
        else:
            cur += " " + w
    if cur:
        lines.append(cur.strip())
    return "\n".join(lines)

def _risk_label(d):
    if d < 0.15: return "低 LOW RISK"
    if d < 0.35: return "中 MEDIUM RISK"
    return "高 HIGH RISK"

def _risk_color(d):
    if d < 0.15: return PALETTE["solid_bright"]
    if d < 0.35: return PALETTE["uncertain_bright"]
    return PALETTE["confab_bright"]

def _point_color(c):
    if c >= 0.80: return PALETTE["solid_bright"]
    if c >= 0.60: return PALETTE["uncertain_bright"]
    return PALETTE["confab_bright"]


# ── CSS — full mystical aesthetic ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@300;400;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

:root {
    --bg:      #0A0A14;
    --mist:    #12122A;
    --panel:   #0F0F20;
    --border:  #2A2A4A;
    --gold:    #C9A84C;
    --gold2:   #7A6030;
    --ghost:   #B8B8CC;
    --text:    #D0D0E8;
    --dim:     #505070;
    --solid:   #6BAA7A;
    --warn:    #C9A030;
    --danger:  #C04040;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

/* ── Animated fog layers ── */
.gradio-container::before,
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 200%; height: 100%;
    pointer-events: none;
    z-index: 0;
    background: radial-gradient(ellipse 80% 40% at 20% 60%,
        rgba(40, 40, 90, 0.18) 0%, transparent 70%),
        radial-gradient(ellipse 60% 30% at 80% 30%,
        rgba(30, 30, 70, 0.14) 0%, transparent 70%);
    animation: fog1 28s ease-in-out infinite alternate;
}

.gradio-container::after {
    background: radial-gradient(ellipse 70% 50% at 60% 80%,
        rgba(60, 40, 100, 0.10) 0%, transparent 70%),
        radial-gradient(ellipse 50% 35% at 10% 20%,
        rgba(20, 20, 60, 0.12) 0%, transparent 70%);
    animation: fog2 35s ease-in-out infinite alternate;
    opacity: 0.7;
}

@keyframes fog1 {
    0%   { transform: translateX(0) translateY(0) scale(1); opacity: 0.6; }
    50%  { transform: translateX(-5%) translateY(2%) scale(1.03); opacity: 0.9; }
    100% { transform: translateX(5%) translateY(-2%) scale(0.98); opacity: 0.5; }
}
@keyframes fog2 {
    0%   { transform: translateX(0) translateY(0); opacity: 0.5; }
    100% { transform: translateX(3%) translateY(3%); opacity: 0.8; }
}

/* ── Layout ── */
.gradio-container > * { position: relative; z-index: 1; }
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* ── Header / hero ── */
#mayoiga-header {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    position: relative;
}

/* Torii gate SVG sits here via HTML component */
#torii-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 1.2rem;
    opacity: 0.75;
    filter: drop-shadow(0 0 18px rgba(201,168,76,0.25));
    animation: torii-breathe 8s ease-in-out infinite;
}

@keyframes torii-breathe {
    0%, 100% { opacity: 0.65; filter: drop-shadow(0 0 14px rgba(201,168,76,0.2)); }
    50%       { opacity: 0.85; filter: drop-shadow(0 0 28px rgba(201,168,76,0.4)); }
}

#kanji-title {
    font-family: 'Noto Serif JP', serif !important;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 0.15em;
    text-shadow: 0 0 40px rgba(201,168,76,0.4), 0 0 80px rgba(201,168,76,0.15);
    margin: 0;
    line-height: 1.2;
}

#en-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--dim);
    letter-spacing: 0.35em;
    text-transform: uppercase;
    margin: 0.4rem 0 0.8rem;
}

#tagline {
    font-family: 'Noto Serif JP', serif;
    font-style: italic;
    font-size: 0.88rem;
    color: var(--ghost);
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.9;
    opacity: 0.8;
}

/* ── Divider ── */
.mori-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.5rem 0;
    opacity: 0.4;
}
.mori-divider::before,
.mori-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}
.mori-divider span {
    color: var(--gold);
    font-size: 0.7rem;
    letter-spacing: 0.4em;
    white-space: nowrap;
}

/* ── Inputs ── */
label, .svelte-1gfkn6j { color: var(--ghost) !important; font-size: 0.78rem !important; letter-spacing: 0.08em; }

textarea, input[type="text"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
    transition: border-color 0.3s;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--gold2) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.08) !important;
}

/* ── Slider ── */
input[type="range"] { accent-color: var(--gold); }

/* ── Button ── */
button.primary {
    background: transparent !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    border-radius: 3px !important;
    cursor: pointer;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}
button.primary::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--gold);
    opacity: 0;
    transition: opacity 0.3s;
}
button.primary:hover {
    color: var(--bg) !important;
    box-shadow: 0 0 20px rgba(201,168,76,0.3) !important;
}
button.primary:hover::before { opacity: 1; }
button.primary span { position: relative; z-index: 1; }

/* ── Panels / outputs ── */
.output-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem;
}

/* ── Plots ── */
.plot-container, canvas {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Markdown output ── */
.prose, .markdown-body, [data-testid="markdown"] {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.8 !important;
    background: transparent !important;
}
.prose h2, .markdown-body h2 {
    font-family: 'Noto Serif JP', serif !important;
    color: var(--gold) !important;
    font-size: 1rem !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-top: 1.4rem;
}
.prose h3, .markdown-body h3 {
    color: var(--ghost) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em;
    margin-top: 1rem;
}
.prose code, .markdown-body code {
    background: rgba(201,168,76,0.08) !important;
    color: var(--gold) !important;
    padding: 0.1em 0.35em !important;
    border-radius: 3px !important;
    font-size: 0.8rem !important;
}
.prose hr, .markdown-body hr {
    border-color: var(--border) !important;
    opacity: 0.5;
}

/* ── Examples ── */
.examples-holder { background: transparent !important; }
.example { 
    background: var(--panel) !important; 
    border: 1px solid var(--border) !important;
    color: var(--ghost) !important;
    font-size: 0.75rem !important;
    border-radius: 3px !important;
    transition: border-color 0.2s;
}
.example:hover { border-color: var(--gold2) !important; }

/* ── Footer ── */
#mori-footer {
    text-align: center;
    padding: 2rem 1rem;
    font-size: 0.7rem;
    color: var(--dim);
    letter-spacing: 0.1em;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
    opacity: 0.6;
}
"""

TORII_SVG = """
<div id="torii-wrap">
<svg width="160" height="110" viewBox="0 0 160 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Main crossbeam top (curved) -->
  <path d="M10 32 Q80 14 150 32" stroke="#C9A84C" stroke-width="7" stroke-linecap="round" fill="none"/>
  <!-- Second crossbeam -->
  <line x1="18" y1="48" x2="142" y2="48" stroke="#C9A84C" stroke-width="4.5" stroke-linecap="round"/>
  <!-- Left pillar -->
  <line x1="34" y1="44" x2="34" y2="108" stroke="#C9A84C" stroke-width="6" stroke-linecap="round"/>
  <!-- Right pillar -->
  <line x1="126" y1="44" x2="126" y2="108" stroke="#C9A84C" stroke-width="6" stroke-linecap="round"/>
  <!-- Top crossbeam caps -->
  <line x1="4" y1="32" x2="156" y2="32" stroke="#C9A84C" stroke-width="3" stroke-linecap="round" opacity="0.6"/>
  <!-- Subtle glow orb -->
  <circle cx="80" cy="30" r="3" fill="#C9A84C" opacity="0.5"/>
</svg>
</div>
"""

HEADER_HTML = f"""
<div id="mayoiga-header">
  {TORII_SVG}
  <p id="kanji-title">迷い家</p>
  <p id="en-title">Mayoiga · The Phantom House · Project 01 of Mayoi no Mori</p>
  <p id="tagline">
        Deep in the mountains, the lost soul treads the arcane ways. <br>
        Weary eyes find lamplight, fire lit, table set with grace. <br>
        Dawn arrives - no walls, no warmth, no trace it ever stood in that frost. <br>
        This is Mayoiga. The phantom born from the ache of being lost
  </p>
</div>
"""

DIVIDER_HTML = """
<div class="mori-divider"><span>✦  ✦  ✦</span></div>
"""

FOOTER_HTML = """
<div id="mori-footer">
  迷いの森 · Mayoi no Mori · A Computational Study of Networks That Lose Themselves<br>
  Powered by Groq · Llama 3.3 · sentence-transformers · Open Source · MIT License
</div>
"""


# ── Gradio app ────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="迷い家 — Mayoiga") as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(DIVIDER_HTML)

    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="問い — Enter your question",
                placeholder="Ask about a real person, an invented researcher, an obscure event…",
                lines=2,
            )
        with gr.Column(scale=1):
            n_variants = gr.Slider(
                minimum=3, maximum=7, value=5, step=1,
                label="変形 — Variants",
                info="More = more accurate, slower"
            )

    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=question_input,
        label="試す — Try these (last three are phantom houses)",
    )

    analyze_btn = gr.Button("✦  Enter the Forest  ✦", variant="primary")

    gr.HTML(DIVIDER_HTML)

    with gr.Row():
        heatmap_out = gr.Plot(label="幻灯 — Sentence Risk Map")
        curve_out   = gr.Plot(label="狐火 — Consistency Curve")

    summary_out = gr.Markdown(label="判読 — Reading")

    analyze_btn.click(
        fn=run_detection,
        inputs=[question_input, n_variants],
        outputs=[heatmap_out, curve_out, summary_out],
    )

    gr.HTML(DIVIDER_HTML)
    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch()