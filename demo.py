"""
Demo - Run the pipeline with sample vishing and non-vishing texts
"""

from utils.zsl_labeler import ZSLLabeler
from models.bert_classifier import VishingBERTClassifier

ZSL_CANDIDATE_LABELS = [
    "urgent financial request",
    "bank account verification",
    "OTP or PIN request",
    "personal identity verification",
    "threatening or pressuring language",
    "prize or lottery scam",
    "impersonating authority or government",
    "normal conversation",
    "technical support scam",
    "loan or credit card offer",
]

SAMPLE_TEXTS = [
    {
        "label": "VISHING - Bank Scam",
        "text": (
            "Hello, this is the fraud department of your bank. We have detected "
            "unauthorized transactions on your account. To prevent your account from "
            "being suspended, please verify your identity by providing your OTP and "
            "account number immediately. This is urgent."
        )
    },
    {
        "label": "VISHING - IRS Scam",
        "text": (
            "This is a final notice from the IRS. A lawsuit has been filed against you. "
            "You owe back taxes and must pay via gift card or wire transfer immediately "
            "or face arrest. Call us back now to avoid legal action."
        )
    },
    {
        "label": "VISHING - Tech Support Scam",
        "text": (
            "Hi, I'm calling from Microsoft technical support. We've detected a virus "
            "on your computer. I need remote access to fix it. Please give me your "
            "Windows password so we can secure your system right away."
        )
    },
    {
        "label": "SAFE - Normal Greeting",
        "text": "Hi how are you? I am fine thank you. Hope you are having a good day."
    },
    {
        "label": "SAFE - Dentist Reminder",
        "text": (
            "Hey, it's Sarah from the dentist's office. Just calling to remind you "
            "about your appointment tomorrow at 3 PM. Please call us back if you need "
            "to reschedule. Have a great day!"
        )
    },
    {
        "label": "SAFE - Delivery Update",
        "text": (
            "Good afternoon, this is James from ABC logistics. I'm calling regarding "
            "your shipment order number 12345. It's been dispatched and should arrive "
            "by Thursday. Let me know if you have any questions."
        )
    }
]


def run_demo():
    print("\n" + "="*65)
    print("   VISHING DETECTOR - DEMO MODE")
    print("   Vishing threshold: >70% confidence")
    print("="*65)

    zsl = ZSLLabeler()
    bert = VishingBERTClassifier()

    for i, sample in enumerate(SAMPLE_TEXTS, 1):
        print(f"\n{'─'*65}")
        print(f"  Sample {i}: {sample['label']}")
        print(f"  Text: \"{sample['text'][:80]}...\"")

        zsl_results = zsl.label(sample["text"], ZSL_CANDIDATE_LABELS)
        top_labels = [label for label, score in zsl_results[:3]]
        zsl_scores_dict = dict(zsl_results)

        print("\n  ZSL Top Labels:")
        for label, score in zsl_results[:3]:
            print(f"    → {label:<42} ({score:.2f})")

        result = bert.predict(sample["text"], context_labels=top_labels, zsl_scores=zsl_scores_dict)

        verdict = "⚠️  VISHING DETECTED" if result["is_vishing"] else "✅  SAFE"
        print(f"\n  VERDICT: {verdict}  |  Confidence: {result['confidence']:.1%}  |  Risk: {result['risk_level']}")
        print(f"  Scores → BERT: {result['scores']['bert']:.2f} | Keywords: {result['scores']['keywords']:.2f} | ZSL: {result['scores']['zsl']:.2f} | Penalty: -{result['scores']['penalty']:.2f}")

    print(f"\n{'='*65}\n")
