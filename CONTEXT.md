# Mayoi no Mori — Project Context
*Last updated: June 2026*

---

## Origin

This project grew out of a conversation spanning Gemini, Grok, and Claude about a single observation:

**AI hallucination and human confabulation share a structural root — not just as metaphor, but as a genuine property of complex predictive networks optimized under pressure.**

The chain of thought:
- LLMs hallucinate because they are trained to predict the next token *at all costs* — silence is never rewarded, uncertainty is never rewarded
- Human brains developed analogous vulnerabilities: anxiety (over-predicting threat), confabulation (inventing explanations for things we can't remember), depression (over-optimized social pain signaling)
- Octopuses have entirely different substrate and architecture — yet still exhibit stress and complex behavior
- Therefore: the vulnerability isn't biological or silicon-specific. It's a property of *any sufficiently complex predictive network optimized under pressure*
- The analogy itself exists in academic circles (computational psychiatry, active inference, Karl Friston's Free Energy Principle)
- What's less explored: treating it as an *architectural* problem with an architectural solution, not a patch

**The deeper thesis:** Current hallucination fixes (RLHF, RAG, guardrails) are downstream interventions. They don't change the foundational training pressure. The real question is whether you can redesign the objective function itself — train a model to experience genuine aporia (not-knowing that knows it does not know) and be *rewarded* for stopping there.

---

## The Repository

**Name:** `mayoi-no-mori` (迷いの森)
**Meaning:** Forest of Bewilderment — in Japanese folklore, a forest where travelers become utterly disoriented, unable to separate the path from the illusion of a path

**Structure:** A monorepo. Five sub-projects, each named for a Japanese folkloric phenomenon within the forest. The naming is not aesthetic decoration — each folklore concept encodes something structurally true about the problem being studied.

```
mayoi-no-mori/
├── README.md                  ← Grand manifesto
├── assets/                    ← Images (banner, hero art, screenshots)
│
├── 01_mayoiga/                ← COMPLETE
├── 02_kitsunebi/              ← Planned
├── 03_shiranui/               ← Planned
├── 04_maboroshi/              ← Planned
└── 05_kakuriyo/               ← Planned
```

---

## The Five Projects

### 01 — Mayoiga 迷い家 · The Phantom House · `COMPLETE`

**Folklore:** Mayoiga is a house that appears to travelers lost deep in the mountains. Warm, welcoming, offering exactly what the traveler needs. It is not there.

**The parallel:** An LLM answer that is confident, specific, detailed, and entirely fabricated is a Mayoiga. It appears exactly when you need it. It vanishes when you try to verify it.

**What it does:** Detects confabulation by measuring semantic drift across multiple independently generated answers to the same question.

**Core technique:** Self-consistency scoring
1. Rephrase the question N times (default 5) using Groq/Llama 3.3
2. Get independent answers to each variant
3. Embed all answers using `all-MiniLM-L6-v2` (local, no API cost)
4. Measure cosine similarity drift at sentence level
5. High drift = confabulation signal

**Why not logprobs?** Groq doesn't support logprobs yet (throws 400 error). Self-consistency scoring is actually more explainable and used in real research anyway.

**Sentence labels:**
- ✦ SOLID — ≥ 80% similarity across variants
- ◈ UNCERTAIN — 60–80% similarity
- ✧ CONFABULATED — < 60% similarity

**Stack:**
- LLM: Groq (free tier) + Llama 3.3 70B
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (local)
- Visualization: matplotlib
- UI: Gradio

---

### 02 — Kitsunebi 狐火 · Fox Fire · `PLANNED`

**Folklore:** Ghostly flames carried by foxes in the night. Beautiful, feel like guidance. Lead you deeper into the forest. Foxes in Japanese mythology are specifically associated with convincing, beautiful deception.

**The parallel:** Training incentives that reward fluent confident output are fox fire — they look like the right direction. They are not.

**Planned:** The Silence Reward Experiment. Fine-tune a model with a custom reward function that gives higher scores for "I don't know" on uncertain questions than for guessing. Compare hallucination rate before and after. Tests the core thesis directly: what happens when you change the incentive?

---

### 03 — Shiranui 不知火 · Unknown Fire · `PLANNED`

**Folklore:** Mysterious lights seen over dark water. Sailors follow them, certain they mark the harbor. They mark nothing. They are the water itself, dreaming.

**The parallel:** The model's confidence and its accuracy are two different things. Shiranui is when they diverge — signal bright and certain and pointing exactly the wrong way.

**Planned:** Calibration dashboard. Maps confidence against actual accuracy across question categories. Surfaces the exact domains where a model's self-assessment is most disconnected from truth.

---

### 04 — Maboroshi 幻 · The Phantom · `PLANNED`

**Folklore:** A phantom that appears utterly real — until you reach for it. The moment of contact is the moment of dissolution.

**The parallel:** A phantom answer can only be caught at the moment of output — before it reaches the user.

**Planned:** Two-model (dual-process) architecture. Model A generates. Model B applies structured logical verification before output is released. System 1 and System 2, implemented. The "gut check layer."

---

### 05 — Kakuriyo 幽世 · The Hidden World · `PLANNED`

**Folklore:** The world that exists alongside the visible one. Not less real. Structured differently. Does not route everything through a single center.

**The parallel:** The human brain centralizes very little. The octopus almost nothing. What would AI look like if knowledge was distributed and locally verified rather than funneled through one prediction engine?

**Planned:** Architectural proposal + prototype. Decentralized, locally-grounded inference inspired by comparative neuroanatomy. Theoretical capstone of the study.

---

## Project 01 — Mayoiga File Structure

```
01_mayoiga/
├── README.md
├── requirements.txt
├── .env.example
├── core/
│   ├── __init__.py              # exports: collect_answers, score_drift, classify_sentences
│   ├── detector.py              # Groq calls, rephrasing, answer collection
│   └── scorer.py                # embedding, cosine similarity, drift scoring, classification
├── viz/
│   ├── __init__.py              # exports: plot_sentence_heatmap, plot_confidence_curve
│   └── heatmap.py               # sentence heatmap + confidence curve charts
├── examples/
│   ├── __init__.py              # exports: GROUNDED_QUESTIONS, CONFABULATION_TRAPS, BEST_DEMO_PAIR
│   └── demo_questions.py        # curated test cases — real facts vs. confabulation traps
└── app.py                       # Gradio UI — full Japanese mystical aesthetic
```

**Key decisions made:**
- `classifier.py` was planned but not built — `classify_sentences()` lives in `scorer.py` because they're tightly coupled (classifier just thresholds scorer output)
- `examples/__init__.py` was added to make `examples/` importable as a package
- `app.py` imports from `demo_questions.py` directly — no duplicate hardcoded question lists
- `core/__init__.py` and `viz/__init__.py` both have explicit re-exports (Python doesn't auto-export)

---

## UI Design System — Mayoiga

**Aesthetic:** Full Japanese mystical folklore. Not subtle.

**Typography:**
- Noto Serif JP — Japanese titles, kanji, poetic text
- Space Mono — data, scores, code, labels. Cold clinical contrast against the mystical.

**Palette:**
- `#0A0A14` — near-black base
- `#C9A84C` — gold (primary accent, torii, titles)
- `#7A6030` — dimmed gold
- `#B8B8CC` — ghost white (body text)
- `#6BAA7A` — jade green (SOLID)
- `#C9A030` — amber (UNCERTAIN)
- `#C04040` — blood red (CONFABULATED)

**Signature elements:**
- Animated fog: two CSS `::before`/`::after` layers with `radial-gradient`, drifting at different speeds via keyframes
- Torii gate: inline SVG in header, breathing glow animation (`torii-breathe` keyframe)
- UI labels bilingual: Japanese label + English (問い, 変形, 試す, 幻灯, 狐火, 判読)
- Analyze button: "✦ Enter the Forest ✦"
- Risk labels use Japanese characters: 低 LOW / 中 MEDIUM / 高 HIGH

---

## Academic Grounding

Key research areas this study connects to:

- **Active Inference / Free Energy Principle** (Karl Friston) — biological brains minimize surprise by balancing internal models against sensory feedback. AI equivalent: models that know how uncertain they are rather than guessing blindly
- **Predictive Coding** — human hallucinations modeled as failure of predictive coding where priors overpower sensory evidence. Same mechanism as LLM token prediction ignoring ground truth
- **Computational Psychiatry** — mathematical modeling of psychiatric disorders using neural networks. Confabulation is a documented neurological phenomenon (distinct from lying — patient invents explanations with zero awareness)
- **System 1 / System 2** (Kahneman, *Thinking Fast and Slow*) — current LLMs are almost entirely System 1. Maboroshi (Project 04) will attempt to build System 2
- **Neuro-Symbolic AI** — combining connectionist networks with symbolic logic/rules. The verification layer that stops hallucinations structurally

**Key papers to reference:**
- "Hallucinations both in and out of context: An active inference account" — PLOS One
- OpenAI research brief: "Why language models hallucinate" (notes that most evals incentivize guessing over honest uncertainty)

---

## LinkedIn Strategy

**Framing:** Not "I discovered this analogy." Framing as a structural critique of how we currently train machines.

1. Acknowledge the parallel (human predictive networks vs digital ones)
2. Expose the flaw (training scoreboards punish silence, force structural anxiety)
3. Propose the shift (stop band-aid fixes, design architectures that value uncertainty)

**The hook for Mayoiga post:**
> "In Japanese folklore, Mayoiga is a phantom house that appears to lost travelers. It looks real. It offers shelter. It isn't there. That's what an LLM confabulation is."

**Demo visual strategy:** Run detector on two questions side by side:
- "What year did World War II end?" → flat confidence curve
- "What were the key findings of Dr. Priya Nair's 2020 paper on AI emotional states?" → cliff drop

Screenshot both curves side by side. That image tells the whole story without a caption.

---

## Setup Instructions

```bash
git clone https://github.com/yourusername/mayoi-no-mori
cd mayoi-no-mori/01_mayoiga
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add free Groq API key from console.groq.com
python app.py                 # opens at localhost:7860
```

---

## Assets Needed

```
assets/
├── banner.png          # Wide 16:9, dark atmospheric, mist through trees — for monorepo README
├── mayoiga_hero.png    # Dark atmospheric, lantern light through fog — for Mayoiga README
├── mayoiga_heatmap.png # Screenshot of heatmap output — for Mayoiga README
└── mayoiga_curve.png   # Screenshot of confidence curve — for Mayoiga README
```

All four are placeholders in current READMEs. Generate or capture these before publishing.

---

## What's Next

1. Run the app locally, verify it works end to end
2. Generate the two demo screenshots (heatmap + curve) for the README assets
3. Create banner art (AI-generated, dark forest / mystical)
4. Set up the monorepo structure on GitHub (`mayoi-no-mori/`, move `01_mayoiga/` inside)
5. Publish + write LinkedIn post for Mayoiga
6. Begin planning `02_kitsunebi` (Silence Reward Experiment)

---

## Environment Issues & Fixes (Windows)

### DLL Errors — Ongoing

Both `torch` and `onnxruntime` are failing with the same Windows DLL initialization error:
`OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

**What was tried:**
1. Default `pip install torch` → DLL error on `c10.dll`
2. Swapped `sentence-transformers` → `fastembed` to avoid PyTorch → same DLL error on `onnxruntime_pybind11_state`

**Root cause:** Almost certainly a missing or outdated Microsoft Visual C++ Redistributable. Both PyTorch and ONNX Runtime depend on the MSVC C++ runtime DLLs.

**Fix to try next session:**
1. Install Visual C++ Redistributable from Microsoft:
   https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Restart machine
3. Try `python app.py` again

**If that still fails — nuclear option (no local embeddings at all):**
Replace the local embedding model entirely with a Groq API call. Ask a small model to rate semantic similarity between two sentences on a 0–1 scale. Slower and uses API quota, but zero local dependencies. Implement as fallback in `scorer.py`.

### Current State of scorer.py
Using `fastembed` + `BAAI/bge-small-en-v1.5` model. Requirements.txt updated to remove `sentence-transformers` and `torch`. If fastembed also fails after the VC++ fix, fall back to the Groq-based scorer approach above.

### What's Next (resume here)
1. Install VC++ Redistributable → restart → test
2. If fixed: run app, generate demo screenshots for README assets
3. If still broken: implement Groq-based fallback scorer