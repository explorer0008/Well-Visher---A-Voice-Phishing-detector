import re
from typing import Dict
def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)          # remove punctuation
    text = re.sub(r"\s+", " ", text)              # collapse spaces
    return text
def _wer(reference: str, hypothesis: str) -> Dict:
    """
    Word Error Rate calculation.
    WER = (S + D + I) / N  where N = number of reference words.
    Returns accuracy = (1 - WER) * 100 clamped to [0, 100].
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return {"wer": 0.0, "accuracy": 100.0, "errors": 0, "total": 0,
                "substitutions": 0, "deletions": 0, "insertions": 0}

    n = len(ref_words)
    m = len(hyp_words)

    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(m + 1):
        d[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost, s, dl, ins = d[i - 1][j - 1]
                d[i][j] = (cost, s, dl, ins)
            else:
                sub_cost  = d[i - 1][j - 1][0] + 1
                del_cost  = d[i - 1][j][0] + 1
                ins_cost  = d[i][j - 1][0] + 1
                best = min(sub_cost, del_cost, ins_cost)
                if best == sub_cost:
                    prev = d[i - 1][j - 1]
                    d[i][j] = (best, prev[1] + 1, prev[2], prev[3])
                elif best == del_cost:
                    prev = d[i - 1][j]
                    d[i][j] = (best, prev[1], prev[2] + 1, prev[3])
                else:
                    prev = d[i][j - 1]
                    d[i][j] = (best, prev[1], prev[2], prev[3] + 1)

    total_errors, subs, dels, ins = d[n][m]
    wer = total_errors / n
    accuracy = max(0.0, (1.0 - wer) * 100)

    return {
        "wer": round(wer, 4),
        "accuracy": round(accuracy, 2),
        "errors": total_errors,
        "total": n,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
    }


def _grade(accuracy: float) -> tuple:
   
    if accuracy >= 85:
        return "Excellent ", "#4ade80"
    elif accuracy >= 65:
        return "Good ", "#fbbf24"
    elif accuracy >= 40:
        return "Fair ", "#fb923c"
    else:
        return "Poor ", "#f87171"


def _similarity(t1: str, t2: str) -> float:
    
    w1 = set(_normalise(t1).split())
    w2 = set(_normalise(t2).split())
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def compare_transcripts(
    reference: str,
    whisper_text: str,
    whisperx_text: str,
    vosk_text: str,
) -> Dict:
    
    ref_norm = _normalise(reference)

    engines = {
        "Faster-Whisper": whisper_text,
        "WhisperX":       whisperx_text,
        "Vosk":           vosk_text,
    }

    results = {}
    for name, text in engines.items():
      
        is_error = text.startswith("[")
        if is_error:
            results[name] = {
                "transcript": text,
                "available": False,
                "accuracy": None,
                "wer": None,
                "errors": None,
                "total": None,
                "substitutions": None,
                "deletions": None,
                "insertions": None,
                "grade": ("N/A", "#94a3b8"),
            }
        else:
            hyp_norm = _normalise(text)
            stats = _wer(ref_norm, hyp_norm)
            grade, color = _grade(stats["accuracy"])
            results[name] = {
                "transcript": text,
                "available": True,
                "accuracy": stats["accuracy"],
                "wer": stats["wer"],
                "errors": stats["errors"],
                "total": stats["total"],
                "substitutions": stats["substitutions"],
                "deletions": stats["deletions"],
                "insertions": stats["insertions"],
                "grade": (grade, color),
            }
    available_texts = {k: v["transcript"] for k, v in results.items() if v["available"]}
    similarity_scores = {}
    engine_names = list(available_texts.keys())
    for i, a in enumerate(engine_names):
        for b in engine_names[i + 1:]:
            key = f"{a} ↔ {b}"
            similarity_scores[key] = round(_similarity(available_texts[a], available_texts[b]) * 100, 1)
    valid_accuracies = [v["accuracy"] for v in results.values() if v["accuracy"] is not None]
    consensus = round(sum(valid_accuracies) / len(valid_accuracies), 2) if valid_accuracies else None
    consensus_grade, consensus_color = _grade(consensus) if consensus is not None else ("N/A", "#94a3b8")
    high_agreement = all(s >= 70.0 for s in similarity_scores.values()) if similarity_scores else False

    return {
        "engines": results,
        "cross_similarity": similarity_scores,
        "consensus_accuracy": consensus,
        "consensus_grade": (consensus_grade, consensus_color),
        "high_agreement": high_agreement,
    }
