"""
Vishing Detection System - GUI v5 with Multi-Engine Accuracy Comparison
Step 3 now runs Faster-Whisper, WhisperX and Vosk side-by-side and
reports per-engine WER accuracy + cross-engine similarity.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

from utils.transcriber import transcribe_audio, transcribe_whisperx, transcribe_vosk
from utils.zsl_labeler import ZSLLabeler
from utils.accuracy_comparator import compare_transcripts
from models.bert_classifier import VishingBERTClassifier

SAMPLE_RATE = 16000
MAX_DURATION = 45
AUDIO_PATH = "data/recorded_audio.wav"

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

zsl_model = None
bert_model = None


def preload_models():
    global zsl_model, bert_model
    zsl_model = ZSLLabeler()
    bert_model = VishingBERTClassifier()


# ─────────────────────────────────────────────────────────────────
class VishingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vishing Detection System")
        self.root.geometry("820x820")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        self.is_recording = False
        self.audio_frames = []
        self.stream = None
        self.elapsed = 0
        self.transcript = ""          # Faster-Whisper transcript (primary)
        self.whisperx_transcript = "" # WhisperX transcript
        self.vosk_transcript = ""     # Vosk transcript

        self._build_scrollable_ui()
        self.log("⏳ Loading AI models in background...")
        threading.Thread(target=self._preload, daemon=True).start()

    # ── Model preload ──────────────────────────────────────────────
    def _preload(self):
        preload_models()
        self.log("✅ Models loaded and ready!")
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    # ── Scrollable shell ──────────────────────────────────────────
    def _build_scrollable_ui(self):
        BG = "#1e1e2e"
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(outer, orient="vertical", bg="#2a2a3e")
        scrollbar.pack(side="right", fill="y")

        self.canvas = tk.Canvas(outer, bg=BG,
                                yscrollcommand=scrollbar.set,
                                highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.canvas.yview)

        self.inner = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        self._build_content(self.inner)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── UI content ────────────────────────────────────────────────
    def _build_content(self, parent):
        CARD   = "#2a2a3e"
        TEXT   = "#e2e8f0"
        MUTED  = "#94a3b8"
        ACCENT = "#7c3aed"
        BG     = "#1e1e2e"

        # Title
        tk.Label(parent, text="🛡️ Vishing Detection System",
                 font=("Segoe UI", 18, "bold"), bg=BG, fg=TEXT).pack(pady=(18, 2))
        tk.Label(parent, text="Real-time voice phishing detector",
                 font=("Segoe UI", 10), bg=BG, fg=MUTED).pack(pady=(0, 10))

        # ── STEP 1 ────────────────────────────────────────────────
        self._section_label(parent, "STEP 1 — Record Audio")
        ctrl = tk.Frame(parent, bg=CARD, padx=20, pady=12)
        ctrl.pack(fill="x", padx=20, pady=(0, 6))

        btn_row = tk.Frame(ctrl, bg=CARD)
        btn_row.pack(fill="x", pady=(4, 6))

        self.start_btn = tk.Button(
            btn_row, text="🎙️  Start Recording",
            font=("Segoe UI", 11, "bold"), bg="#16a34a", fg="white",
            relief="flat", padx=18, pady=7,
            command=self.start_recording, state=tk.DISABLED)
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = tk.Button(
            btn_row, text="⏹️  Stop Recording",
            font=("Segoe UI", 11, "bold"), bg="#dc2626", fg="white",
            relief="flat", padx=18, pady=7,
            command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side="left")

        self.timer_label = tk.Label(ctrl, text="⏱️  00:00 / 00:45",
            font=("Segoe UI", 10), bg=CARD, fg=MUTED)
        self.timer_label.pack(anchor="w", pady=(2, 2))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor="#13131f", background="#16a34a")
        self.progress = ttk.Progressbar(
            ctrl, maximum=MAX_DURATION,
            style="green.Horizontal.TProgressbar")
        self.progress.pack(fill="x")

        self.status_label = tk.Label(ctrl, text="● Idle — Press Start to begin",
            font=("Segoe UI", 10), bg=CARD, fg=MUTED)
        self.status_label.pack(anchor="w", pady=(5, 0))

        # ── STEP 2 ────────────────────────────────────────────────
        self._section_label(parent, "STEP 2 — Transcription  (Faster-Whisper)")
        trans = tk.Frame(parent, bg=CARD, padx=20, pady=10)
        trans.pack(fill="x", padx=20, pady=(0, 6))

        self.transcript_box = scrolledtext.ScrolledText(
            trans, height=4, font=("Segoe UI", 10),
            bg="#13131f", fg=TEXT, relief="flat", wrap="word")
        self.transcript_box.pack(fill="x")
        self.transcript_box.insert(
            tk.END, "Transcript will appear here after recording...")
        self.transcript_box.config(state=tk.DISABLED)

        # ── STEP 3 — Multi-engine Accuracy ───────────────────────
        self._section_label(parent,
            "STEP 3 — Transcription Accuracy Comparison  ")
        acc = tk.Frame(parent, bg=CARD, padx=20, pady=12)
        acc.pack(fill="x", padx=20, pady=(0, 6))

        tk.Label(acc,
            text="Type what you actually said — then click Compare to check all three engines:",
            font=("Segoe UI", 9), bg=CARD, fg=MUTED).pack(anchor="w", pady=(0, 4))

        self.reference_box = tk.Text(acc, height=3, font=("Segoe UI", 10),
            bg="#13131f", fg=TEXT, relief="flat", wrap="word")
        self.reference_box.pack(fill="x")

        acc_row = tk.Frame(acc, bg=CARD)
        acc_row.pack(fill="x", pady=(8, 6))

        self.check_btn = tk.Button(
            acc_row, text="📊  Compare All Engines",
            font=("Segoe UI", 10, "bold"), bg="#0369a1", fg="white",
            relief="flat", padx=14, pady=5,
            command=self.check_accuracy, state=tk.DISABLED)
        self.check_btn.pack(side="left")

        tk.Label(acc_row,
            
            font=("Segoe UI", 9), bg=CARD, fg=MUTED).pack(side="left")

        # Engine result cards
        self.engine_frame = tk.Frame(acc, bg=CARD)
        self.engine_frame.pack(fill="x", pady=(4, 0))

        # --- Faster-Whisper card
        self.fw_card  = self._engine_card(self.engine_frame, "⚡ Faster-Whisper")
        self.wx_card  = self._engine_card(self.engine_frame, "🔬 WhisperX")
        self.vsk_card = self._engine_card(self.engine_frame, "🗣️  Vosk")

        # Similarity + consensus row
        self.similarity_label = tk.Label(acc, text="",
            font=("Segoe UI", 9), bg=CARD, fg=MUTED, justify="left")
        self.similarity_label.pack(anchor="w", pady=(6, 0))

        self.consensus_label = tk.Label(acc, text="",
            font=("Segoe UI", 11, "bold"), bg=CARD, fg=TEXT)
        self.consensus_label.pack(anchor="w", pady=(2, 0))

        # ── STEP 4 ────────────────────────────────────────────────
        self._section_label(parent, "STEP 4 — Vishing Classification")
        classify_frame = tk.Frame(parent, bg=CARD, padx=20, pady=14)
        classify_frame.pack(fill="x", padx=20, pady=(0, 6))

        self.proceed_btn = tk.Button(
            classify_frame, text="🚀  Proceed to Classify",
            font=("Segoe UI", 13, "bold"), bg=ACCENT, fg="white",
            activebackground="#6d28d9", relief="flat", padx=24, pady=10,
            command=self.run_classification, state=tk.DISABLED)
        self.proceed_btn.pack(anchor="w")

        tk.Label(classify_frame,
            text="Runs ZSL + BERT hybrid pipeline on the Faster-Whisper transcript",
            font=("Segoe UI", 9), bg=CARD, fg=MUTED).pack(anchor="w", pady=(4, 0))

        # Verdict area
        self.verdict_label = tk.Label(classify_frame, text="",
            font=("Segoe UI", 20, "bold"), bg=CARD, fg=TEXT)
        self.verdict_label.pack(anchor="w", pady=(14, 2))

        self.confidence_label = tk.Label(classify_frame, text="",
            font=("Segoe UI", 10), bg=CARD, fg=MUTED, justify="left")
        self.confidence_label.pack(anchor="w")

        self.indicators_label = tk.Label(classify_frame, text="",
            font=("Segoe UI", 10), bg=CARD, fg="#fbbf24", wraplength=720,
            justify="left")
        self.indicators_label.pack(anchor="w", pady=(4, 0))

        self.zsl_label = tk.Label(classify_frame, text="",
            font=("Segoe UI", 9), bg=CARD, fg=MUTED,
            wraplength=720, justify="left")
        self.zsl_label.pack(anchor="w", pady=(4, 0))

        # ── Log console ───────────────────────────────────────────
        self._section_label(parent, "Console Log")
        log_frame = tk.Frame(parent, bg=CARD, padx=20, pady=10)
        log_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.log_box = scrolledtext.ScrolledText(
            log_frame, height=8, font=("Consolas", 9),
            bg="#13131f", fg="#94a3b8", relief="flat", wrap="word")
        self.log_box.pack(fill="x")

    # ── Engine result card helper ──────────────────────────────────
    def _engine_card(self, parent, title: str) -> dict:
        CARD  = "#2a2a3e"
        INNER = "#1a1a2e"
        TEXT  = "#e2e8f0"
        MUTED = "#94a3b8"

        frame = tk.Frame(parent, bg=INNER, padx=12, pady=8)
        frame.pack(fill="x", pady=(0, 4))

        title_lbl = tk.Label(frame, text=title,
            font=("Segoe UI", 10, "bold"), bg=INNER, fg=TEXT)
        title_lbl.pack(anchor="w")

        accuracy_lbl = tk.Label(frame, text="—",
            font=("Segoe UI", 13, "bold"), bg=INNER, fg=MUTED)
        accuracy_lbl.pack(anchor="w")

        detail_lbl = tk.Label(frame, text="",
            font=("Segoe UI", 9), bg=INNER, fg=MUTED)
        detail_lbl.pack(anchor="w")

        transcript_lbl = tk.Label(frame, text="",
            font=("Segoe UI", 9, "italic"), bg=INNER, fg=MUTED,
            wraplength=700, justify="left")
        transcript_lbl.pack(anchor="w")

        return {
            "accuracy": accuracy_lbl,
            "detail":   detail_lbl,
            "transcript": transcript_lbl,
        }

    # ── Section header helper ─────────────────────────────────────
    def _section_label(self, parent, text: str):
        BG = "#1e1e2e"
        tk.Label(parent, text=f"  {text}",
                 font=("Segoe UI", 10, "bold"),
                 bg=BG, fg="#7c3aed").pack(
                     fill="x", padx=20, pady=(10, 2))

    # ── Log ───────────────────────────────────────────────────────
    def log(self, message: str):
        def _do():
            self.log_box.insert(tk.END, message + "\n")
            self.log_box.see(tk.END)
        self.root.after(0, _do)

    # ── Recording ─────────────────────────────────────────────────
    def start_recording(self):
        self.audio_frames = []
        self.elapsed = 0
        self.is_recording = True
        self.transcript = ""
        self.whisperx_transcript = ""
        self.vosk_transcript = ""

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.proceed_btn.config(state=tk.DISABLED)
        self.check_btn.config(state=tk.DISABLED)
        self.status_label.config(text="🔴 Recording...", fg="#f87171")

        self.verdict_label.config(text="")
        self.confidence_label.config(text="")
        self.indicators_label.config(text="")
        self.zsl_label.config(text="")
        self._reset_engine_cards()

        self.transcript_box.config(state=tk.NORMAL)
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.insert(tk.END, "Recording in progress...")
        self.transcript_box.config(state=tk.DISABLED)

        self.log(f"🎙️ Recording started (max {MAX_DURATION}s)...")

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype=np.int16, callback=self._audio_callback)
        self.stream.start()
        threading.Thread(target=self._run_timer, daemon=True).start()

    def _reset_engine_cards(self):
        MUTED = "#94a3b8"
        for card in (self.fw_card, self.wx_card, self.vsk_card):
            card["accuracy"].config(text="—", fg=MUTED)
            card["detail"].config(text="")
            card["transcript"].config(text="")
        self.similarity_label.config(text="")
        self.consensus_label.config(text="")

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_frames.append(indata.copy())

    def _run_timer(self):
        while self.is_recording and self.elapsed < MAX_DURATION:
            time.sleep(0.5)
            self.elapsed += 0.5
            e = self.elapsed
            m, s = int(e) // 60, int(e) % 60
            self.root.after(0, lambda m=m, s=s, e=e: (
                self.timer_label.config(text=f"⏱️  {m:02d}:{s:02d} / 00:45"),
                self.progress.config(value=e)
            ))
        if self.is_recording:
            self.root.after(0, self.stop_recording)

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏳ Transcribing audio...", fg="#fbbf24")
        self.log(f"⏹️ Stopped at {int(self.elapsed)}s. Transcribing...")

        os.makedirs("data", exist_ok=True)
        if self.audio_frames:
            audio_data = np.concatenate(self.audio_frames, axis=0)
            wav.write(AUDIO_PATH, SAMPLE_RATE, audio_data)
            threading.Thread(target=self._transcribe_all, daemon=True).start()
        else:
            self.log("⚠️ No audio captured.")
            self.start_btn.config(state=tk.NORMAL)

    # ── Transcription (all engines) ───────────────────────────────
    def _transcribe_all(self):
        """Run primary Faster-Whisper transcription; others are run only on demand."""
        try:
            self.log("⚡ Faster-Whisper transcribing...")
            self.transcript = transcribe_audio(AUDIO_PATH)
            self.log(f'   ✅ Whisper: "{self.transcript[:80]}"')

            self.root.after(0, lambda: (
                self.transcript_box.config(state=tk.NORMAL),
                self.transcript_box.delete("1.0", tk.END),
                self.transcript_box.insert(tk.END, self.transcript),
                self.transcript_box.config(state=tk.DISABLED),
                self.check_btn.config(state=tk.NORMAL),
                self.proceed_btn.config(state=tk.NORMAL),
                self.start_btn.config(state=tk.NORMAL),
                self.status_label.config(
                    text="✅ Transcript ready — Compare accuracy (optional) or Proceed to Classify",
                    fg="#4ade80")
            ))
        except Exception as e:
            self.log(f"❌ Transcription error: {str(e)}")
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    # ── Accuracy comparison ───────────────────────────────────────
    def check_accuracy(self):
        reference = self.reference_box.get("1.0", tk.END).strip()
        if not self.transcript.strip() or not reference:
            messagebox.showwarning(
                "Missing Text",
                "Both a transcript and your reference text are required.")
            return

        self.check_btn.config(state=tk.DISABLED, text="⏳ Running all engines...")
        self._reset_engine_cards()
        self.log("📊 Running multi-engine accuracy comparison...")

        threading.Thread(
            target=self._run_accuracy_comparison,
            args=(reference,),
            daemon=True
        ).start()

    def _run_accuracy_comparison(self, reference: str):
        # Run WhisperX and Vosk on the saved audio
        self.log("🔬 WhisperX transcribing...")
        self.whisperx_transcript = transcribe_whisperx(AUDIO_PATH)
        self.log(f'   WhisperX: "{self.whisperx_transcript[:80]}"')

        self.log("🗣️  Vosk transcribing...")
        self.vosk_transcript = transcribe_vosk(AUDIO_PATH)
        self.log(f'   Vosk: "{self.vosk_transcript[:80]}"')

        # Compare
        report = compare_transcripts(
            reference,
            self.transcript,
            self.whisperx_transcript,
            self.vosk_transcript,
        )
        self.root.after(0, lambda: self._display_accuracy(report))

    def _display_accuracy(self, report: dict):
        cards = {
            "Faster-Whisper": self.fw_card,
            "WhisperX":       self.wx_card,
            "Vosk":           self.vsk_card,
        }

        for engine_name, card in cards.items():
            data = report["engines"][engine_name]
            if not data["available"]:
                card["accuracy"].config(text="Not available", fg="#94a3b8")
                # Show the error message
                short = data["transcript"][:90] + ("…" if len(data["transcript"]) > 90 else "")
                card["transcript"].config(text=short)
                card["detail"].config(text="")
            else:
                grade_text, color = data["grade"]
                card["accuracy"].config(
                    text=f"{data['accuracy']:.1f}%  —  {grade_text}",
                    fg=color)
                card["detail"].config(
                    text=(f"WER: {data['wer']:.3f}  |  "
                          f"Errors: {data['errors']}/{data['total']} words  |  "
                          f"Sub: {data['substitutions']}  "
                          f"Del: {data['deletions']}  "
                          f"Ins: {data['insertions']}"),
                    fg="#94a3b8")
                preview = data["transcript"][:100] + ("…" if len(data["transcript"]) > 100 else "")
                card["transcript"].config(text=f'"{preview}"', fg="#64748b")
                self.log(f"   {engine_name}: {data['accuracy']:.1f}%  ({grade_text})")

        # Cross-engine similarity
        if report["cross_similarity"]:
            sim_lines = "  ".join(
                f"{pair}: {score:.0f}%"
                for pair, score in report["cross_similarity"].items()
            )
            agreement = "🟢 High agreement across engines" if report["high_agreement"] \
                        else "🟡 Engines show some variation"
            self.similarity_label.config(
                text=f"Cross-engine similarity — {sim_lines}\n{agreement}")

        # Consensus
        if report["consensus_accuracy"] is not None:
            grade_text, color = report["consensus_grade"]
            self.consensus_label.config(
                text=f"Consensus Accuracy: {report['consensus_accuracy']:.1f}%  ({grade_text})",
                fg=color)
            self.log(f"📊 Consensus accuracy: {report['consensus_accuracy']:.1f}%")

        self.check_btn.config(state=tk.NORMAL, text="📊  Compare All Engines")

    # ── Classification ────────────────────────────────────────────
    def run_classification(self):
        if not self.transcript.strip():
            messagebox.showwarning("No Transcript", "Please record audio first.")
            return

        self.proceed_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.verdict_label.config(text="⏳ Classifying...", fg="#fbbf24")
        self.confidence_label.config(text="")
        self.indicators_label.config(text="")
        self.zsl_label.config(text="")
        self.log("🚀 Running classification pipeline...")
        threading.Thread(target=self._classify, daemon=True).start()

    def _classify(self):
        try:
            self.log("🏷️ Running ZSL labeling...")
            zsl_results = zsl_model.label(self.transcript, ZSL_CANDIDATE_LABELS)
            top_labels = [label for label, score in zsl_results[:3]]
            zsl_scores_dict = dict(zsl_results)
            zsl_display = "ZSL Labels:  " + "   |   ".join(
                [f"{l} ({s:.2f})" for l, s in zsl_results[:3]]
            )
            for label, score in zsl_results[:3]:
                self.log(f"   → {label} ({score:.2f})")

            self.log("🤖 Running BERT classification...")
            result = bert_model.predict(
                self.transcript,
                context_labels=top_labels,
                zsl_scores=zsl_scores_dict)

            self.root.after(0, lambda: self._show_verdict(result, zsl_display))

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            self.root.after(0, lambda: (
                self.proceed_btn.config(state=tk.NORMAL),
                self.start_btn.config(state=tk.NORMAL)
            ))

    def _show_verdict(self, result, zsl_display):
        conf = result["confidence"]

        if result["is_vishing"]:
            self.verdict_label.config(text="⚠️  VISHING DETECTED", fg="#f87171")
        else:
            self.verdict_label.config(text="✅  CALL APPEARS SAFE", fg="#4ade80")

        self.confidence_label.config(
            text=(
                f"Confidence: {conf:.1%}   |   Risk Level: {result['risk_level']}\n"
                f"BERT: {result['scores']['bert']:.2f}   "
                f"Keywords: {result['scores']['keywords']:.2f}   "
                f"ZSL: {result['scores']['zsl']:.2f}   "
                f"Safe Penalty: -{result['scores']['penalty']:.2f}   "
                f"Final: {result['scores']['final']:.2f}"
            ),
            fg="#94a3b8")
        self.indicators_label.config(
            text=f"Key Indicators: {', '.join(result['indicators'])}")
        self.zsl_label.config(text=zsl_display)
        self.proceed_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.NORMAL)
        self.status_label.config(text="● Classification complete", fg="#4ade80")

        self.root.after(100, lambda: self.canvas.yview_moveto(1.0))

        self.log(
            f"{'⚠️ VISHING DETECTED' if result['is_vishing'] else '✅ SAFE'} "
            f"— Confidence: {conf:.1%}  Risk: {result['risk_level']}"
        )


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = VishingApp(root)
    root.mainloop()
