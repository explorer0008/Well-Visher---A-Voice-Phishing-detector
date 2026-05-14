"""
XAI Explainer — Explainable AI for the Vishing Detection System
================================================================
Provides three layers of explanation:

1. TOKEN-LEVEL ATTRIBUTION  (LIME-style perturbation)
   Each word in the transcript is masked and the change in confidence
   is measured → words are ranked by how much they push/pull the score.

2. SCORE DECOMPOSITION  (SHAP-inspired feature contributions)
   Breaks the final confidence score into labelled slices:
   keyword contribution, ZSL contribution, BERT sentiment, safe penalty.

3. NATURAL-LANGUAGE EXPLANATION  (rule + data driven)
   Generates a plain-English paragraph that explains WHY the system
   reached its verdict, what drove the score, and what the user should
   do next.

All methods are deterministic (no sampling) and run on CPU in < 2 s.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any


# ── helpers ──────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Split transcript into word tokens, preserving order."""
    return re.findall(r"\b\w+(?:'\w+)?\b", text)


def _mask_token(tokens: List[str], idx: int) -> str:
    """Return transcript with the token at *idx* replaced by a blank."""
    masked = tokens[:]
    masked[idx] = ""
    return " ".join(t for t in masked if t)


# ── Token-level attribution ───────────────────────────────────────────────────

def token_attribution(
    transcript: str,
    bert_classifier,          # VishingBERTClassifier instance
    context_labels: List[str],
    zsl_scores: Dict[str, float],
    base_confidence: float,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Compute per-token importance by leave-one-out perturbation.

    For every word w_i:
        importance(w_i) = base_confidence − confidence(transcript without w_i)

    A positive importance means removing the word *lowers* the vishing score
    (i.e. the word drives suspicion up).
    A negative importance means removing the word *raises* the score
    (i.e. the word is a safe / moderating signal).

    Returns a list of dicts sorted by |importance|, capped at *top_n*.
    """
    tokens = _tokenise(transcript)
    if not tokens:
        return []

    results: List[Dict[str, Any]] = []

    for i, token in enumerate(tokens):
        masked_text = _mask_token(tokens, i)
        if not masked_text.strip():
            continue
        try:
            pred = bert_classifier.predict(
                masked_text,
                context_labels=context_labels,
                zsl_scores=zsl_scores,
            )
            importance = base_confidence - pred["confidence"]
        except Exception:
            importance = 0.0

        results.append({
            "token": token,
            "importance": round(importance, 4),
            "direction": "risk" if importance > 0 else "safe",
        })

    # Sort by absolute importance, descending
    results.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return results[:top_n]


# ── Score decomposition ───────────────────────────────────────────────────────

def score_decomposition(scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Return an ordered list of labelled contribution slices that sum to the
    final score (accounting for the safe penalty deduction).

    Each slice has:
        label       — human-readable component name
        value       — signed contribution to the final score
        pct         — percentage of |final| that this component contributes
        color       — suggested hex colour for bar charts
    """
    kw   = scores.get("keywords", 0.0) * 0.50   # 50 % weight
    zsl  = scores.get("zsl", 0.0)     * 0.30   # 30 % weight
    bert = scores.get("bert", 0.0)    * 0.20   # 20 % weight
    pen  = -scores.get("penalty", 0.0)          # negative contribution

    final = scores.get("final", kw + zsl + bert + pen)
    total = abs(kw) + abs(zsl) + abs(bert) + abs(pen) or 1.0

    slices = [
        {"label": "Keyword Match",      "value": round(kw,   4), "color": "#f87171"},
        {"label": "ZSL Context",        "value": round(zsl,  4), "color": "#fb923c"},
        {"label": "BERT Sentiment",     "value": round(bert, 4), "color": "#fbbf24"},
        {"label": "Safe-word Penalty",  "value": round(pen,  4), "color": "#4ade80"},
    ]

    for s in slices:
        s["pct"] = round(abs(s["value"]) / total * 100, 1)

    return slices


# ── Natural-language explanation ──────────────────────────────────────────────

_RISK_TEMPLATES = {
    "high": [
        "The system flagged this call as a likely vishing attempt with {conf:.0%} confidence (risk level: {risk}).",
        "The primary driver was {top_driver}.",
        "{keyword_sentence}",
        "{zsl_sentence}",
        "The BERT sentiment analysis also detected negative / coercive language (score {bert:.2f}).",
        "{action}",
    ],
    "medium": [
        "The call raised moderate suspicion (confidence {conf:.0%}, risk: {risk}).",
        "Some vishing-like patterns were detected — {top_driver} was the strongest signal.",
        "{keyword_sentence}",
        "{zsl_sentence}",
        "{action}",
    ],
    "low": [
        "The call appears safe (confidence {conf:.0%}, risk: {risk}).",
        "No strong vishing keywords or suspicious context labels were found.",
        "{safe_sentence}",
        "The BERT model rated the sentiment as mostly neutral / positive (score {bert:.2f}).",
        "{action}",
    ],
}

_ACTIONS = {
    "high": (
        "⚠️  Recommended action: Hang up immediately. Do NOT share any personal "
        "information, OTPs, or financial details. Report the number to your bank "
        "or local cybercrime authority."
    ),
    "medium": (
        "🟡 Recommended action: Exercise caution. Verify the caller's identity "
        "through an official channel before sharing any sensitive information."
    ),
    "low": (
        "✅ Recommended action: The call appears legitimate. Continue the "
        "conversation normally, but stay alert if unexpected requests arise."
    ),
}


def natural_language_explanation(
    result: Dict[str, Any],
    zsl_results: List[Tuple[str, float]],
    top_tokens: List[Dict[str, Any]],
) -> str:
    """
    Compose a plain-English paragraph explaining the verdict.

    Parameters
    ----------
    result      : dict returned by VishingBERTClassifier.predict()
    zsl_results : list of (label, score) from ZSLLabeler.label()
    top_tokens  : list of dicts from token_attribution()
    """
    conf   = result["confidence"]
    risk   = result["risk_level"]
    scores = result["scores"]

    # Determine risk tier
    if conf >= 0.70:
        tier = "high"
    elif conf >= 0.40:
        tier = "medium"
    else:
        tier = "low"

    # Identify top score driver
    driver_map = {
        "Keyword Match":  scores.get("keywords", 0) * 0.50,
        "ZSL Context":    scores.get("zsl", 0)      * 0.30,
        "BERT Sentiment": scores.get("bert", 0)      * 0.20,
    }
    top_driver = max(driver_map, key=driver_map.get)

    # Keyword sentence
    kw_hits = result.get("indicators", [])
    real_kw  = [k for k in kw_hits if k != "No strong indicators found"]
    if real_kw:
        keyword_sentence = (
            f"High-risk keywords detected included: "
            f"{', '.join(repr(k) for k in real_kw[:4])}. "
        )
    else:
        keyword_sentence = "No high-risk keywords were directly matched. "

    # ZSL sentence
    risk_zsl = [(l, s) for l, s in zsl_results if s >= 0.40 and l != "normal conversation"]
    if risk_zsl:
        top_zsl = risk_zsl[0]
        zsl_sentence = (
            f"Zero-shot classification identified the context as "
            f"'{top_zsl[0]}' with {top_zsl[1]:.0%} confidence. "
        )
    else:
        zsl_sentence = ""

    # Safe-word sentence (for low-risk only)
    penalty = scores.get("penalty", 0)
    if penalty > 0:
        safe_sentence = (
            f"Safe-context words (e.g. appointment, delivery, greetings) "
            f"reduced the raw score by {penalty:.2f} points. "
        )
    else:
        safe_sentence = ""

    # Top suspicious token (for high/medium)
    risk_tokens = [t for t in top_tokens if t["direction"] == "risk" and t["importance"] > 0.02]
    if risk_tokens:
        top_tok = risk_tokens[0]["token"]
        token_note = f"The word '{top_tok}' had the largest individual impact on the risk score. "
    else:
        token_note = ""

    # Fill template
    template_lines = _RISK_TEMPLATES[tier]
    formatted_lines = []
    for line in template_lines:
        formatted_line = line.format(
            conf=conf,
            risk=risk,
            top_driver=top_driver,
            keyword_sentence=keyword_sentence,
            zsl_sentence=zsl_sentence,
            safe_sentence=safe_sentence,
            bert=scores.get("bert", 0),
            action=_ACTIONS[tier],
        )
        if formatted_line.strip():  # Skip empty lines
            formatted_lines.append(formatted_line)

    # Insert token_note before the action for high/medium risk
    if token_note and tier in ["high", "medium"]:
        action_index = len(formatted_lines) - 1
        formatted_lines.insert(action_index, token_note.strip())

    explanation = "\n".join(f"• {line}" for line in formatted_lines)

    return explanation.strip()


# ── Unified explain() entry point ─────────────────────────────────────────────

def explain(
    transcript: str,
    result: Dict[str, Any],
    zsl_results: List[Tuple[str, float]],
    bert_classifier,
    context_labels: List[str],
    zsl_scores: Dict[str, float],
    run_token_attribution: bool = True,
) -> Dict[str, Any]:
    """
    Master explainability function.  Returns a dict containing:

        tokens          — per-token attribution list
        decomposition   — score slice list
        explanation     — plain-English string
        summary_table   — compact dict for GUI display
    """
    base_conf = result["confidence"]

    # 1. Token attribution (optional, slow on long transcripts)
    if run_token_attribution:
        tokens = token_attribution(
            transcript, bert_classifier,
            context_labels, zsl_scores, base_conf,
        )
    else:
        tokens = []

    # 2. Score decomposition
    decomp = score_decomposition(result["scores"])

    # 3. Natural-language explanation
    nl_explanation = natural_language_explanation(result, zsl_results, tokens)

    # 4. Summary table (for GUI)
    summary = {
        "verdict":     "⚠️ VISHING" if result["is_vishing"] else "✅ SAFE",
        "confidence":  f"{base_conf:.1%}",
        "risk_level":  result["risk_level"],
        "top_risk_words": [t["token"] for t in tokens if t["direction"] == "risk"][:5],
        "top_safe_words": [t["token"] for t in tokens if t["direction"] == "safe"][:3],
        "top_driver":  max(
            {"Keyword": result["scores"]["keywords"] * 0.50,
             "ZSL":     result["scores"]["zsl"]      * 0.30,
             "BERT":    result["scores"]["bert"]      * 0.20},
            key=lambda k: {"Keyword": result["scores"]["keywords"] * 0.50,
                           "ZSL":     result["scores"]["zsl"]      * 0.30,
                           "BERT":    result["scores"]["bert"]      * 0.20}[k]
        ),
    }

    return {
        "tokens":        tokens,
        "decomposition": decomp,
        "explanation":   nl_explanation,
        "summary":       summary,
    }
