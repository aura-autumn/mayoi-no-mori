# 迷いの森 — Mayoi no Mori

> *In Japanese folklore, Mayoi no Mori is a forest where travelers become utterly disoriented —*
> *unable to separate the path from the illusion of a path.*

**Every complex network has a forest it can get lost in. This is a map of that forest.**

---

<!-- BANNER IMAGE -->
<!-- Replace with your forest/mystical banner art -->
<!-- Recommended: wide 16:9, dark atmospheric, mist through trees -->
![Mayoi no Mori](assets/banner.png)

---

## What This Is

This repository is a five-part computational study into a single question:

**Why do complex predictive networks — biological or silicon — confabulate with confidence?**

Humans do it. LLMs do it. The mechanism is structurally similar in both: a system optimized to predict and generate *at all costs* will manufacture output rather than produce silence. It will build a phantom house rather than admit it is lost.

The five projects in this monorepo each investigate a different facet of this problem. Each is named for a phenomenon from Japanese folklore — not for aesthetics, but because these stories were encoding something true about the nature of illusion long before we had the mathematics to describe it.

This is not a collection of hallucination patches. It is an attempt to understand the architecture of the disease before prescribing a cure.

---

## The Forest

```
mayoi-no-mori/
│
├── 01_mayoiga/       迷い家      The Phantom House
├── 02_kitsunebi/     狐火        Fox Fire  
├── 03_shiranui/      不知火      Unknown Fire
├── 04_maboroshi/     幻          The Phantom
└── 05_kakuriyo/      幽世        The Hidden World
```

---

### 01 — Mayoiga 迷い家 · *The Phantom House*

> *Mayoiga is a house that appears to travelers lost deep in the mountains.*
> *It is warm. It is welcoming. It offers everything a lost person needs.*
> *It is not there.*

**Status:** `Complete`

An AI answer that sounds confident, specific, and structured — but is entirely fabricated — is a Mayoiga. It appears exactly when you need it. It offers exactly what you were looking for. It vanishes when you try to verify it.

**What it does:** Detects confabulation in LLM responses by measuring semantic drift across multiple independently generated answers to the same question. Real knowledge produces consistency. Fabrication produces variation.

**Core technique:** Self-consistency scoring via sentence-transformer embeddings + cosine similarity drift analysis.

→ [Enter the Phantom House](./01-mayoiga/)

---

<!-- PROJECT SCREENSHOT -->
<!-- Replace with a screenshot of Mayoiga UI showing the heatmap -->
![Mayoiga Screenshot](assets/mayoiga_demo.png)

---

### 02 — Kitsunebi 狐火 · *Fox Fire*

> *Kitsunebi are the ghostly flames carried by foxes in the night.*
> *They are beautiful. They feel like guidance.*
> *They lead you deeper into the forest.*

**Status:** `Planned`

Foxes in Japanese mythology are specifically associated with *convincing, beautiful deception*. Kitsunebi is the light they carry — it looks like a lantern, like safety, like the right direction. It is none of these things.

**What it will do:** Investigate how training incentives create a structural pressure toward confident output over honest uncertainty. The Silence Reward Experiment — what happens when you fine-tune a model to value "I don't know" over a risky guess?

→ *Coming soon*

---

### 03 — Shiranui 不知火 · *Unknown Fire*

> *Shiranui are mysterious lights seen over dark water.*
> *Sailors follow them, certain they mark the harbor.*
> *They mark nothing. They are the water itself, dreaming.*

**Status:** `Planned`

The model's confidence score and its accuracy are two different things. Shiranui is what happens when they diverge — when the signal is bright and certain and pointing you exactly the wrong way.

**What it will do:** A calibration dashboard that maps confidence against actual accuracy across question categories. Surfaces the exact domains where a model's self-assessment is most disconnected from truth.

→ *Coming soon*

---

### 04 — Maboroshi 幻 · *The Phantom*

> *Maboroshi is a phantom that appears utterly real — until you reach for it.*
> *The moment of contact is the moment of dissolution.*

**Status:** `Planned`

A phantom answer can only be caught at the moment of output — before it reaches the user. Maboroshi is the interceptor: a two-model architecture where one mind generates and a second, skeptical mind catches phantoms before they escape.

**What it will do:** A neuro-symbolic dual-process architecture. Model A generates. Model B applies structured logical verification before any output is released. System 1 and System 2, implemented.

→ *Coming soon*

---

### 05 — Kakuriyo 幽世 · *The Hidden World*

> *Kakuriyo is the world that exists alongside the visible one.*
> *It is not less real. It is simply structured differently.*
> *It does not route everything through a single center.*

**Status:** `Planned`

The human brain centralizes very little. The octopus centralizes almost nothing — most of its processing lives in its arms, local and independent. What would an AI architecture look like if knowledge was distributed and locally verified rather than funneled through one prediction engine?

**What it will do:** An architectural proposal and prototype exploring decentralized, locally-grounded inference — inspired by comparative neuroanatomy. The theoretical capstone of the study.

→ *Coming soon*

---

## The Thesis

Current approaches to fixing hallucination are downstream interventions:

- **RLHF** — punish the model after it lies
- **RAG** — give it a textbook to copy from
- **Guardrails** — catch the output after it's already wrong

None of these change the foundational pressure. The model is still optimized to generate at all costs. It is still lost in the forest. We are just trying to catch it before it reaches the traveler.

The deeper question — the one this study is working toward — is whether you can redesign the objective function itself. Whether a network can be trained to experience *genuine aporia*: the state of not-knowing that knows it does not know, and stops there, rather than building a phantom house to hide inside.

---

## Background

This project grew out of a conversation about whether AI hallucination and human confabulation share a structural root — not just as metaphor, but as a genuine property of complex predictive networks optimized under pressure.

The observation isn't new. Computational psychiatry has been modeling psychosis as predictive coding failure for years. What's less explored is the solution space: if the disease is architectural, the cure should be too.

---

## Stack

All projects in this repository are fully open source. No paid APIs required beyond a free Groq account.

| Layer | Technology |
|---|---|
| LLM Inference | [Groq](https://groq.com) — Llama 3.3 70B (free tier) |
| Embeddings | [Voyage AI](https://voyageai.com) — `voyage-4-lite` (200M free tokens) |
| UI | [Gradio](https://gradio.app) |
| Visualization | matplotlib, seaborn |
| Language | Python 3.10+ |

---

## Getting Started

Each sub-project has its own `README.md` and `requirements.txt`. Start with Mayoiga:

```bash
git clone https://github.com/yourusername/mayoi-no-mori
cd mayoi-no-mori/01_mayoiga
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your free Groq API key
python app.py
```

---

## A Note on the Names

The folklore names are not decoration. Each phenomenon from Japanese tradition was chosen because it encodes something structurally true about the problem being studied — something the story understood intuitively that we are only now formalizing mathematically.

The forest is real. The house in the forest is not. This repository is an attempt to tell the difference.

---

*Built with curiosity about what it means for any mind — biological or silicon — to know something, versus to perform knowing.*