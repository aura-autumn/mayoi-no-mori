"""
detector.py

Sends N rephrased variations of a question to Groq and collects answers.
The key insight: real knowledge produces consistent answers across rephrasings.
Confabulation produces drift — the model makes different things up each time.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

REPHRASE_PROMPT = """You are a question rephraser. Given a question, produce {n} 
semantically identical but differently worded versions of it. 
Return ONLY the questions, one per line, no numbering, no explanation."""

ANSWER_PROMPT = """Answer the following question as accurately as you can.
If you are not sure about something, say so clearly.
Be specific — include dates, names, and numbers where relevant.

Question: {question}"""


def rephrase_question(question: str, n: int = 5) -> list[str]:
    """Generate N rephrased versions of the input question using Groq."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": REPHRASE_PROMPT.format(n=n) + f"\n\nOriginal question: {question}"
            }
        ],
        temperature=0.7,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()
    variants = [q.strip() for q in raw.split("\n") if q.strip()]
    # Always include the original
    return [question] + variants[:n - 1]


def get_answer(question: str, temperature: float = 0.7) -> str:
    """Get a single answer from Groq for a given question."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(question=question)
            }
        ],
        temperature=temperature,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def collect_answers(question: str, n_variants: int = 5) -> dict:
    """
    Full pipeline: rephrase the question N times, get an answer for each variant.
    
    Returns:
        {
            "original_question": str,
            "variants": [str, ...],
            "answers": [str, ...],
        }
    """
    print(f"[detector] Rephrasing question into {n_variants} variants...")
    variants = rephrase_question(question, n=n_variants)

    answers = []
    for i, variant in enumerate(variants):
        print(f"[detector] Getting answer {i + 1}/{len(variants)}...")
        answer = get_answer(variant)
        answers.append(answer)

    return {
        "original_question": question,
        "variants": variants,
        "answers": answers,
    }