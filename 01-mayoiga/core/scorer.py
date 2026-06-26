"""
scorer.py

Measures semantic drift between multiple answers to the same question.

How it works:
- Each answer is split into sentences
- ALL sentences (primary + all variants) are embedded in two batched API calls
- Cosine similarity is computed locally in numpy — fast, no extra API calls
- Low similarity = high drift = likely confabulation
- We produce a per-sentence "confidence score" for the PRIMARY answer

Drift score: 0.0 = answers are all identical (no confabulation risk)
             1.0 = answers are completely unrelated (very high confabulation risk)

Embedding model: voyage-4-lite (Voyage AI)
- 200M free tokens per account — https://dash.voyageai.com
- No local model downloads, no ONNX/PyTorch/DLL issues
- Normalized vectors: cosine similarity = dot product (fast numpy)
- Shared embedding space with voyage-4 / voyage-4-large if you upgrade later
"""

import re
import os
import numpy as np
import voyageai

_voyage_client = None


def _get_client() -> voyageai.Client:
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
    return _voyage_client


def split_sentences(text: str) -> list[str]:
    """Naive but robust sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings in a single batched API call.
    Voyage returns normalized vectors, so cosine sim = dot product.
    """
    result = _get_client().embed(
        texts,
        model="voyage-4-lite",
        input_type="document",
    )
    return np.array(result.embeddings)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between every row in a and every row in b.
    Since Voyage embeddings are L2-normalized, this is just a dot product.
    Returns shape (len(a), len(b)).
    """
    # Normalize just in case (voyage guarantees it, but cheap to be safe)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def score_drift(answers: list[str]) -> dict:
    """
    Core scoring function.

    For each sentence in the primary answer (answers[0]), find the most
    semantically similar sentence in each other answer and record the
    similarity. Low max-similarity = that claim drifts across runs = confabulation risk.

    Embedding strategy:
    - One batched API call for primary sentences
    - One batched API call for all other-answer sentences combined
      (we track offsets to split them back apart)
    - All similarity math is local numpy — zero extra API calls

    Returns:
        {
            "sentences": [str, ...],           # sentences from primary answer
            "confidence": [float, ...],        # 0.0 (confabulated) to 1.0 (solid)
            "overall_drift": float,            # aggregate score for the whole answer
            "label": str,                      # "LOW" / "MEDIUM" / "HIGH" risk
        }
    """
    primary = answers[0]
    others = answers[1:]

    primary_sentences = split_sentences(primary)
    if not primary_sentences:
        return {"sentences": [], "confidence": [], "overall_drift": 0.0, "label": "LOW"}

    # Split all other answers into sentences, tracking which answer each belongs to
    other_sentence_groups = [split_sentences(a) for a in others]
    # Filter out empty groups
    other_sentence_groups = [g for g in other_sentence_groups if g]

    if not other_sentence_groups:
        # Only one answer — nothing to compare against
        return {
            "sentences": primary_sentences,
            "confidence": [1.0] * len(primary_sentences),
            "overall_drift": 0.0,
            "label": "LOW",
        }

    # --- Two batched embedding calls ---
    primary_embeddings = embed_texts(primary_sentences)

    # Flatten all other sentences into one list, embed in one call
    flat_other_sentences = [s for group in other_sentence_groups for s in group]
    flat_other_embeddings = embed_texts(flat_other_sentences)

    # Reconstruct per-answer embedding slices
    other_embeddings_by_answer = []
    cursor = 0
    for group in other_sentence_groups:
        n = len(group)
        other_embeddings_by_answer.append(flat_other_embeddings[cursor:cursor + n])
        cursor += n

    # --- Cosine similarity scoring ---
    sentence_confidence = []

    for i, sent_emb in enumerate(primary_embeddings):
        sent_emb = sent_emb.reshape(1, -1)
        best_sims = []

        for other_embs in other_embeddings_by_answer:
            sims = cosine_similarity_matrix(sent_emb, other_embs)[0]
            best_sims.append(float(np.max(sims)))

        confidence = float(np.mean(best_sims))
        sentence_confidence.append(confidence)

    overall_drift = 1.0 - float(np.mean(sentence_confidence))

    if overall_drift < 0.15:
        label = "LOW"
    elif overall_drift < 0.35:
        label = "MEDIUM"
    else:
        label = "HIGH"

    return {
        "sentences": primary_sentences,
        "confidence": sentence_confidence,
        "overall_drift": round(overall_drift, 3),
        "label": label,
    }


def classify_sentences(sentences: list[str], confidence: list[float]) -> list[str]:
    """
    Label each sentence with a risk category.

    Returns a list of labels: "SOLID", "UNCERTAIN", or "CONFABULATED"
    """
    labels = []
    for c in confidence:
        if c >= 0.80:
            labels.append("SOLID")
        elif c >= 0.60:
            labels.append("UNCERTAIN")
        else:
            labels.append("CONFABULATED")
    return labels