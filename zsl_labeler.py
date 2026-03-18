"""
ZSL Labeler - Zero-Shot Learning Partial Labeling
"""

from transformers import pipeline
from typing import List, Tuple

ZSL_MODEL = "facebook/bart-large-mnli"

class ZSLLabeler:
    def __init__(self):
        print("  Loading ZSL model (facebook/bart-large-mnli)...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=ZSL_MODEL,
            device=-1
        )
        print("  ZSL model ready ✅")

    def label(
        self,
        text: str,
        candidate_labels: List[str],
        multi_label: bool = True,
        threshold: float = 0.15
    ) -> List[Tuple[str, float]]:
        """
        Partially label text using zero-shot classification.
        """
        if not text or text == "[No speech detected]":
            return [("no speech", 1.0)]

        result = self.classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=multi_label
        )

        paired = list(zip(result["labels"], result["scores"]))
        paired.sort(key=lambda x: x[1], reverse=True)
        filtered = [(label, score) for label, score in paired if score >= threshold]

        return filtered if filtered else paired[:1]
