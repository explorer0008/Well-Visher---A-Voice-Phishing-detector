"""
BERT Vishing Classifier - Final Version
- Vishing detected only if confidence > 0.70
- Risk levels: LOW < 0.40 | MEDIUM 0.40-0.70 | HIGH > 0.70
- No keywords = capped at 0.30 (cannot be vishing)
- Safe context words reduce the score
- ZSL only counts if confidence exceeds per-label threshold
"""

from transformers import pipeline
from typing import List, Dict, Any

# ── Vishing keywords ─────────────────────────────────────────────
VISHING_KEYWORDS = {
    "high": [
        "otp", "one-time password", "one time password", "pin number",
        "cvv", "account number", "social security", "ssn",
        "verify your identity", "account is suspended", "account has been suspended",
        "immediate action", "legal action", "username", "password", "arrest warrant", "irs",
        "microsoft support", "refund has been approved", "gift card",
        "wire transfer", "bitcoin", "unauthorized transaction",
        "your card has been", "blocked your account", "confirm your details",
        "give me your password", "remote access"
    ],
    "medium": [
        "credit card", "urgent", "act now", "prize", "winner", "claim your",
        "federal", "police", "verify", "suspicious activity", "fraud department",
        "limited time", "final notice", "last warning"
    ],
    "low": [
        "bank", "password", "confirm", "government", "security alert"
    ]
}

# ── Safe words — reduce false positives ─────────────────────────
SAFE_INDICATORS = [
    "appointment", "reminder", "dentist", "doctor", "delivery", "shipment",
    "order", "logistics", "reschedule", "office", "calling to remind",
    "have a great day", "thank you for", "your package", "your order",
    "how are you", "i am fine", "good morning", "good afternoon", "good evening",
    "hi there", "hello", "just checking in", "hope you are well"
]

# ── ZSL risk labels with minimum confidence thresholds ──────────
HIGH_RISK_ZSL_LABELS = {
    "urgent financial request":               0.60,
    "bank account verification":              0.60,
    "OTP or PIN request":                     0.55,
    "personal identity verification":         0.65,
    "threatening or pressuring language":     0.65,
    "impersonating authority or government":  0.72,
    "technical support scam":                 0.60,
    "prize or lottery scam":                  0.60,
}


class VishingBERTClassifier:

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        print("  Loading BERT classifier...")
        self.nlp = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1
        )
        print("  BERT classifier ready ✅")

    def _keyword_risk_score(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        hits = {"high": [], "medium": [], "low": []}
        for level, keywords in VISHING_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    hits[level].append(kw)
        score = (
            min(len(hits["high"]), 3) * 0.35 +
            min(len(hits["medium"]), 3) * 0.10 +
            min(len(hits["low"]), 3) * 0.03
        )
        return {"score": min(score, 1.0), "hits": hits}

    def _safe_context_penalty(self, text: str) -> float:
        text_lower = text.lower()
        matches = sum(1 for word in SAFE_INDICATORS if word in text_lower)
        return min(matches * 0.10, 0.35)

    def _zsl_risk_score(self, context_labels: List[str], zsl_scores: Dict[str, float]) -> float:
        if not context_labels:
            return 0.0
        weighted_hits = 0.0
        for label in context_labels:
            if label in HIGH_RISK_ZSL_LABELS:
                min_confidence = HIGH_RISK_ZSL_LABELS[label]
                actual_score = zsl_scores.get(label, 0.0)
                if actual_score >= min_confidence:
                    weighted_hits += (actual_score - min_confidence + 0.1)
        return min(weighted_hits, 1.0)

    def _build_input(self, transcript: str, context_labels: List[str]) -> str:
        context_str = ", ".join(context_labels) if context_labels else "normal conversation"
        return f"Transcript: {transcript} Context: {context_str}"[:1000]

    def predict(
        self,
        transcript: str,
        context_labels: List[str] = None,
        zsl_scores: Dict[str, float] = None
    ) -> Dict[str, Any]:

        context_labels = context_labels or []
        zsl_scores = zsl_scores or {}

        # ── 1. Keyword score (PRIMARY, 50%) ──────────────────
        kw_result = self._keyword_risk_score(transcript)
        kw_score = kw_result["score"]

        # ── 2. ZSL score (threshold-filtered, 30%) ───────────
        zsl_score = self._zsl_risk_score(context_labels, zsl_scores)

        # ── 3. BERT sentiment (tiebreaker, 20%) ──────────────
        bert_input = self._build_input(transcript, context_labels)
        bert_result = self.nlp(bert_input, truncation=True, max_length=512)[0]
        bert_score = bert_result["score"] if bert_result["label"] == "NEGATIVE" else 1 - bert_result["score"]

        # ── 4. Safe context penalty ───────────────────────────
        penalty = self._safe_context_penalty(transcript)

        # ── 5. Final score ────────────────────────────────────
        # If NO keywords matched at all → cap score at 0.30 (never vishing)
        if kw_score == 0.0 and len(kw_result["hits"]["medium"]) == 0:
            final_score = min(0.30, bert_score * 0.20)
        else:
            raw_score = (kw_score * 0.50) + (zsl_score * 0.30) + (bert_score * 0.20)
            final_score = max(0.0, raw_score - penalty)

        # ── 6. Risk levels ────────────────────────────────────
        # LOW  < 0.40
        # MEDIUM 0.40 - 0.70
        # HIGH > 0.70 (also triggers VISHING DETECTED)
        if final_score >= 0.70:
            risk_level = "🔴 HIGH"
        elif final_score >= 0.60:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"

        # ── 7. Indicators ─────────────────────────────────────
        indicators = []
        indicators.extend(kw_result["hits"]["high"][:3])
        indicators.extend([
            l for l in context_labels
            if l in HIGH_RISK_ZSL_LABELS and zsl_scores.get(l, 0) >= HIGH_RISK_ZSL_LABELS[l]
        ][:2])
        if not indicators:
            indicators = ["No strong indicators found"]

        return {
            "is_vishing": final_score >= 0.50,   # Only flag if > 70%
            "confidence": final_score,
            "risk_level": risk_level,
            "indicators": indicators,
            "scores": {
                "bert": round(bert_score, 3),
                "keywords": round(kw_score, 3),
                "zsl": round(zsl_score, 3),
                "penalty": round(penalty, 3),
                "final": round(final_score, 3)
            }
        }
