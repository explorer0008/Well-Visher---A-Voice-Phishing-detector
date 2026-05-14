
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
from utils.xai_explainer import explain
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
class VishingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vishing Detection System — XAI Edition")
        self.root.geometry("860x900")
        self.root.configure(bg="#f5f5f5")
        self.root.resizable(True, True)
        self.is_recording = False
        self.audio_frames = []
        self.stream = None
        self.elapsed = 0
        self.transcript = ""
        self.whisperx_transcript = ""
        self.vosk_transcript = ""
        self._last_result = None
        self._last_zsl_results = []
        self._last_context_labels = []
        self._last_zsl_scores = {}

        self.risk_canvas = None

        self._build_scrollable_ui()
        threading.Thread(target=self._preload, daemon=True).start()

    def _preload(self):
        preload_models()
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    def _build_scrollable_ui(self):
        BG = "#f5f5f5"
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(outer, orient="vertical", bg="#d0d0d0")
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

    def _build_content(self, parent):
        CARD   = "#ffffff"
        TEXT   = "#1a1a1a"
        MUTED  = "#666666"
        ACCENT = "#7c3aed"
        BG     = "#f5f5f5"

        
        tk.Label(parent, text="Vishing Detection System",
                 font=("Segoe UI", 18, "bold"), bg=BG, fg=TEXT).pack(pady=(18, 2))
        tk.Label(parent, text="Real-time voice phishing detector  ·  Explainable AI Edition",
                 font=("Segoe UI", 10), bg=BG, fg=MUTED).pack(pady=(0, 10))

        self._section_label(parent, "STEP 1 — Record Audio")
        ctrl = tk.Frame(parent, bg=CARD, padx=20, pady=12)
        ctrl.pack(fill="x", padx=20, pady=(0, 6))

        btn_row = tk.Frame(ctrl, bg=CARD)
        btn_row.pack(fill="x", pady=(4, 6))

        self.start_btn = tk.Button(
            btn_row, text="Start Recording",
            font=("Segoe UI", 11, "bold"), bg="#16a34a", fg="white",
            relief="flat", padx=18, pady=7,
            command=self.start_recording, state=tk.DISABLED)
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = tk.Button(
            btn_row, text="Stop Recording",
            font=("Segoe UI", 11, "bold"), bg="#dc2626", fg="white",
            relief="flat", padx=18, pady=7,
            command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side="left", padx=(0, 10))

        self.reset_btn = tk.Button(
            btn_row, text="Reset",
            font=("Segoe UI", 11, "bold"), bg="#6b7280", fg="white",
            relief="flat", padx=18, pady=7,
            command=self.reset_session)
        self.reset_btn.pack(side="left")

        self.timer_label = tk.Label(ctrl, text="00:00 / 00:45",
            font=("Segoe UI", 10), bg=CARD, fg=MUTED)
        self.timer_label.pack(anchor="w", pady=(2, 2))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor="#e8e8e8", background="#16a34a")
        self.progress = ttk.Progressbar(
            ctrl, maximum=MAX_DURATION,
            style="green.Horizontal.TProgressbar")
        self.progress.pack(fill="x")

        self.status_label = tk.Label(ctrl, text="● Idle — Press Start to begin",
            font=("Segoe UI", 10), bg=CARD, fg=MUTED)
        self.status_label.pack(anchor="w", pady=(5, 0))

        self._section_label(parent, "STEP 2 — Transcription  (Faster-Whisper)")
        trans = tk.Frame(parent, bg=CARD, padx=20, pady=10)
        trans.pack(fill="x", padx=20, pady=(0, 6))

        self.transcript_box = scrolledtext.ScrolledText(
            trans, height=4, font=("Segoe UI", 10),
            bg="#f9f9f9", fg=TEXT, relief="flat", wrap="word")
        self.transcript_box.pack(fill="x")
        self.transcript_box.insert(
            tk.END, "Transcript will appear here after recording...")
        self.transcript_box.config(state=tk.DISABLED)


        self._section_label(parent, "STEP 3 — Vishing Classification")
        classify_frame = tk.Frame(parent, bg=CARD, padx=20, pady=14)
        classify_frame.pack(fill="x", padx=20, pady=(0, 6))

        self.proceed_btn = tk.Button(
            classify_frame, text="Proceed to Classify",
            font=("Segoe UI", 13, "bold"), bg=ACCENT, fg="white",
            activebackground="#6d28d9", relief="flat", padx=24, pady=10,
            command=self.run_classification, state=tk.DISABLED)
        self.proceed_btn.pack(anchor="w")

        tk.Label(classify_frame,
            text="Runs ZSL + BERT hybrid pipeline on the Faster-Whisper transcript",
            font=("Segoe UI", 9), bg=CARD, fg=MUTED).pack(anchor="w", pady=(4, 0))

        self.verdict_label = tk.Label(classify_frame, text="",
            font=("Segoe UI", 20, "bold"), bg=CARD, fg=TEXT)
        self.verdict_label.pack(anchor="w", pady=(14, 2))

        self.confidence_label = tk.Label(classify_frame, text="",
            font=("Segoe UI", 10), bg=CARD, fg=MUTED, justify="left")
        self.confidence_label.pack(anchor="w")

        self.risk_canvas = tk.Canvas(classify_frame, height=40, bg=CARD, highlightthickness=0)
        self.risk_canvas.pack(fill="x", pady=(4, 0))

        self.indicators_label = tk.Label(classify_frame, text="",
            font=("Segoe UI", 10), bg=CARD, fg="#fbbf24", wraplength=760,
            justify="left")
        self.indicators_label.pack(anchor="w", pady=(4, 0))

        self.zsl_label = tk.Label(classify_frame, text="",
            font=("Segoe UI", 9), bg=CARD, fg=MUTED,
            wraplength=760, justify="left")
        self.zsl_label.pack(anchor="w", pady=(4, 0))

        self._section_label(parent, "STEP 4 — Explainable AI (XAI)")
        xai_frame = tk.Frame(parent, bg=CARD, padx=20, pady=14)
        xai_frame.pack(fill="x", padx=20, pady=(0, 6))

        xai_btn_row = tk.Frame(xai_frame, bg=CARD)
        xai_btn_row.pack(fill="x", pady=(0, 8))

        self.xai_btn = tk.Button(
            xai_btn_row, text="Explain Decision",
            font=("Segoe UI", 11, "bold"), bg="#c026d3", fg="white",
            activebackground="#a21caf", relief="flat", padx=18, pady=7,
            command=self.run_xai, state=tk.DISABLED)
        self.xai_btn.pack(side="left", padx=(0, 10))

        tk.Label(xai_btn_row,
            text="Runs token attribution + score decomposition to explain the verdict",
            font=("Segoe UI", 9), bg=CARD, fg=MUTED).pack(side="left")

        tk.Label(xai_frame, text="Score Decomposition",
            font=("Segoe UI", 10, "bold"), bg=CARD, fg=TEXT).pack(anchor="w")

        self.decomp_canvas = tk.Canvas(xai_frame, height=60, bg="#f9f9f9",
                                       highlightthickness=0)
        self.decomp_canvas.pack(fill="x", pady=(4, 8))

        legend_frame = tk.Frame(xai_frame, bg=CARD)
        legend_frame.pack(anchor="w")
        tk.Label(legend_frame, text="Token Attribution",
            font=("Segoe UI", 10, "bold"), bg=CARD, fg=TEXT).pack(side="left")
        tk.Label(legend_frame, text=" risk-raising ",
            font=("Segoe UI", 9), bg="#ef4444", fg="white", relief="ridge", bd=1).pack(side="left", padx=(8, 2))
        tk.Label(legend_frame, text=" risk-lowering ",
            font=("Segoe UI", 9), bg="#22c55e", fg="white", relief="ridge", bd=1).pack(side="left")

        self.token_frame = tk.Frame(xai_frame, bg="#f9f9f9")
        self.token_frame.pack(fill="x", pady=(4, 8))

        self.token_placeholder = tk.Label(
            self.token_frame,
            text="Token attribution will appear after explanation runs.",
            font=("Segoe UI", 9, "italic"), bg="#f9f9f9", fg="#999999")
        self.token_placeholder.pack(anchor="w", padx=6, pady=4)

        tk.Label(xai_frame, text="Natural-Language Explanation",
            font=("Segoe UI", 10, "bold"), bg=CARD, fg=TEXT).pack(anchor="w")

        self.xai_text = scrolledtext.ScrolledText(
            xai_frame, height=6, font=("Segoe UI", 10),
            bg="#f9f9f9", fg="#1a1a1a", relief="flat", wrap="word",
            state=tk.DISABLED)
        self.xai_text.pack(fill="x", pady=(4, 0))

    def _engine_card(self, parent, title: str) -> dict:
        INNER = "#f9f9f9"
        TEXT  = "#1a1a1a"
        MUTED = "#666666"

        frame = tk.Frame(parent, bg=INNER, padx=12, pady=8)
        frame.pack(fill="x", pady=(0, 4))

        tk.Label(frame, text=title,
            font=("Segoe UI", 10, "bold"), bg=INNER, fg=TEXT).pack(anchor="w")

        accuracy_lbl = tk.Label(frame, text="—",
            font=("Segoe UI", 13, "bold"), bg=INNER, fg=MUTED)
        accuracy_lbl.pack(anchor="w")

        detail_lbl = tk.Label(frame, text="",
            font=("Segoe UI", 9), bg=INNER, fg=MUTED)
        detail_lbl.pack(anchor="w")

        transcript_lbl = tk.Label(frame, text="",
            font=("Segoe UI", 9, "italic"), bg=INNER, fg=MUTED,
            wraplength=740, justify="left")
        transcript_lbl.pack(anchor="w")

        return {"accuracy": accuracy_lbl, "detail": detail_lbl,
                "transcript": transcript_lbl}

    def _section_label(self, parent, text: str):
        tk.Label(parent, text=f"  {text}",
                 font=("Segoe UI", 10, "bold"),
                 bg="#f5f5f5", fg="#7c3aed").pack(
                     fill="x", padx=20, pady=(10, 2))

    def log(self, message: str):
        pass

    def start_recording(self):
        self.audio_frames = []
        self.elapsed = 0
        self.is_recording = True
        self.transcript = ""
        self.whisperx_transcript = ""
        self.vosk_transcript = ""
        self._last_result = None

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.DISABLED)
        self.proceed_btn.config(state=tk.DISABLED)
        self.xai_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Recording...", fg="#f87171")

        for lbl in (self.verdict_label, self.confidence_label,
                    self.indicators_label, self.zsl_label):
            lbl.config(text="")
        self._reset_xai_panel()

        self.transcript_box.config(state=tk.NORMAL)
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.insert(tk.END, "Recording in progress...")
        self.transcript_box.config(state=tk.DISABLED)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype=np.int16, callback=self._audio_callback)
        self.stream.start()
        threading.Thread(target=self._run_timer, daemon=True).start()

    def _reset_xai_panel(self):
        self.decomp_canvas.delete("all")
        for w in self.token_frame.winfo_children():
            w.destroy()
        self.token_placeholder = tk.Label(
            self.token_frame,
            text="Token attribution will appear after explanation runs.",
            font=("Segoe UI", 9, "italic"), bg="#f9f9f9", fg="#999999")
        self.token_placeholder.pack(anchor="w", padx=6, pady=4)
        self.xai_text.config(state=tk.NORMAL)
        self.xai_text.delete("1.0", tk.END)
        self.xai_text.config(state=tk.DISABLED)

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
                self.timer_label.config(text=f"{m:02d}:{s:02d} / 00:45"),
                self.progress.config(value=e)
            ))
        if self.is_recording:
            self.root.after(0, self.stop_recording)

    def reset_session(self):
        if self.is_recording:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

        self.elapsed = 0
        self.audio_frames = []
        self.transcript = ""
        self.whisperx_transcript = ""
        self.vosk_transcript = ""
        self._last_result = None
        self._last_zsl_results = []
        self._last_context_labels = []
        self._last_zsl_scores = {}

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)
        self.proceed_btn.config(state=tk.DISABLED)
        self.xai_btn.config(state=tk.DISABLED, text="Explain Decision")
        self.status_label.config(text="● Idle — Press Start to begin", fg="#666666")
        self.timer_label.config(text="00:00 / 00:45")
        self.progress.config(value=0)

        self.transcript_box.config(state=tk.NORMAL)
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.insert(tk.END, "Transcript will appear here after recording...")
        self.transcript_box.config(state=tk.DISABLED)

        for lbl in (self.verdict_label, self.confidence_label,
                    self.indicators_label, self.zsl_label):
            lbl.config(text="")
        self._reset_xai_panel()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

        self.stop_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Transcribing audio...", fg="#fbbf24")

        os.makedirs("data", exist_ok=True)
        if self.audio_frames:
            audio_data = np.concatenate(self.audio_frames, axis=0)
            wav.write(AUDIO_PATH, SAMPLE_RATE, audio_data)
            threading.Thread(target=self._transcribe_all, daemon=True).start()
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.NORMAL)

    def _transcribe_all(self):
        try:
            self.transcript = transcribe_audio(AUDIO_PATH)

            self.root.after(0, lambda: (
                self.transcript_box.config(state=tk.NORMAL),
                self.transcript_box.delete("1.0", tk.END),
                self.transcript_box.insert(tk.END, self.transcript),
                self.transcript_box.config(state=tk.DISABLED),
                self.proceed_btn.config(state=tk.NORMAL),
                self.start_btn.config(state=tk.NORMAL),
                self.status_label.config(
                    text="Transcript ready — proceed to classify",
                    fg="#4ade80")
            ))
        except Exception as e:
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    def run_classification(self):
        if not self.transcript.strip():
            messagebox.showwarning("No Transcript", "Please record audio first.")
            return
        self.proceed_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.xai_btn.config(state=tk.DISABLED)
        self.verdict_label.config(text="Classifying...", fg="#fbbf24")
        for lbl in (self.confidence_label, self.indicators_label, self.zsl_label):
            lbl.config(text="")
        self.risk_canvas.delete("all")
        self._reset_xai_panel()
        self.log("Running classification pipeline...")
        threading.Thread(target=self._classify, daemon=True).start()

    def _classify(self):
        try:
            zsl_results = zsl_model.label(self.transcript, ZSL_CANDIDATE_LABELS)
            top_labels = [label for label, score in zsl_results[:3]]
            zsl_scores_dict = dict(zsl_results)
            zsl_display = "ZSL Labels:  " + "   |   ".join(
                [f"{l} ({s:.2f})" for l, s in zsl_results[:3]])

            result = bert_model.predict(
                self.transcript,
                context_labels=top_labels,
                zsl_scores=zsl_scores_dict)

            self._last_result = result
            self._last_zsl_results = zsl_results
            self._last_context_labels = top_labels
            self._last_zsl_scores = zsl_scores_dict

            self.root.after(0, lambda: self._show_verdict(result, zsl_display))

        except Exception as e:
            self.root.after(0, lambda: (
                self.proceed_btn.config(state=tk.NORMAL),
                self.start_btn.config(state=tk.NORMAL)
            ))

    def _show_verdict(self, result, zsl_display):
        conf = result["confidence"]
        if result["is_vishing"]:
            self.verdict_label.config(text="VISHING DETECTED", fg="#f87171")
        else:
            self.verdict_label.config(text="CALL APPEARS SAFE", fg="#4ade80")

        self.confidence_label.config(
            text=(f"Confidence: {conf:.1%}   |   Risk Level: {result['risk_level']}\n"
                  f"BERT: {result['scores']['bert']:.2f}   "
                  f"Keywords: {result['scores']['keywords']:.2f}   "
                  f"ZSL: {result['scores']['zsl']:.2f}   "
                  f"Safe Penalty: -{result['scores']['penalty']:.2f}   "
                  f"Final: {result['scores']['final']:.2f}"),
            fg="#666666")
        self.indicators_label.config(
            text=f"Key Indicators: {', '.join(result['indicators'])}")
        self.zsl_label.config(text=zsl_display)
        self._draw_risk_bar(conf)
        self.proceed_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.NORMAL)
        self.xai_btn.config(state=tk.NORMAL)   
        self.reset_btn.config(state=tk.NORMAL)
        self.status_label.config(
            text="● Classification complete — click Explain Decision for XAI",
            fg="#4ade80")
        self.root.after(100, lambda: self.canvas.yview_moveto(0.6))

    def _draw_risk_bar(self, confidence: float):
        """Draw a risk level bar with needle indicating confidence."""
        canvas = self.risk_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        width = canvas.winfo_width() or 760  
        height = 40

        low_color = "#4ade80"
        med_color = "#fbbf24"
        high_color = "#f87171"

        low_end = int(width * 0.4)
        med_end = int(width * 0.7)

        canvas.create_rectangle(0, 10, low_end, height-10, fill=low_color, outline="")
        canvas.create_rectangle(low_end, 10, med_end, height-10, fill=med_color, outline="")
        canvas.create_rectangle(med_end, 10, width, height-10, fill=high_color, outline="")

        
        canvas.create_text(low_end // 2, height // 2, text="LOW", font=("Segoe UI", 10, "bold"), fill="white")
        canvas.create_text((low_end + med_end) // 2, height // 2, text="MEDIUM", font=("Segoe UI", 10, "bold"), fill="black")
        canvas.create_text((med_end + width) // 2, height // 2, text="HIGH", font=("Segoe UI", 10, "bold"), fill="white")

        needle_x = int(confidence * width)
        canvas.create_line(needle_x, 0, needle_x, height, fill="white", width=3)
        canvas.create_polygon(
            needle_x - 5, 0,
            needle_x + 5, 0,
            needle_x, 10,
            fill="white", outline=""
        )

    def run_xai(self):
        if not self._last_result:
            messagebox.showwarning("No Result", "Run classification first.")
            return
        self.xai_btn.config(state=tk.DISABLED, text="Computing XAI...")
        self._reset_xai_panel()
        threading.Thread(target=self._run_xai_thread, daemon=True).start()

    def _run_xai_thread(self):
        try:
            xai_output = explain(
                transcript=self.transcript,
                result=self._last_result,
                zsl_results=self._last_zsl_results,
                bert_classifier=bert_model,
                context_labels=self._last_context_labels,
                zsl_scores=self._last_zsl_scores,
                run_token_attribution=True,
            )
            self.root.after(0, lambda: self._display_xai(xai_output))
        except Exception as e:
            self.root.after(0, lambda: self.xai_btn.config(
                state=tk.NORMAL, text="Explain Decision"))

    def _display_xai(self, xai_output: dict):
        """Render the three XAI components into the UI."""

        self.decomp_canvas.delete("all")
        self.decomp_canvas.update_idletasks()
        total_w = self.decomp_canvas.winfo_width() or 800
        bar_h = 28
        y0 = 16
        slices = xai_output["decomposition"]

        pos_total = sum(s["value"] for s in slices if s["value"] > 0) or 1.0
        neg_total = sum(abs(s["value"]) for s in slices if s["value"] < 0) or 1.0

        x_cursor = 0
        for sl in slices:
            if sl["value"] <= 0:
                continue
            w = int(sl["value"] / pos_total * (total_w * 0.72))
            if w < 2:
                continue
            self.decomp_canvas.create_rectangle(
                x_cursor, y0, x_cursor + w, y0 + bar_h,
                fill=sl["color"], outline="")
            if w > 40:
                self.decomp_canvas.create_text(
                    x_cursor + w // 2, y0 + bar_h // 2,
                    text=f"{sl['label']} {sl['pct']:.0f}%",
                    font=("Segoe UI", 8, "bold"), fill="#ffffff")
            x_cursor += w

        for sl in slices:
            if sl["value"] >= 0:
                continue
            w = int(abs(sl["value"]) / neg_total * (total_w * 0.25))
            if w < 2:
                continue
            px = total_w - w - 2
            self.decomp_canvas.create_rectangle(
                px, y0, px + w, y0 + bar_h, fill="#4ade80", outline="")
            if w > 60:
                self.decomp_canvas.create_text(
                    px + w // 2, y0 + bar_h // 2,
                    text=f"Safe Penalty -{sl['pct']:.0f}%",
                    font=("Segoe UI", 8, "bold"), fill="#ffffff")

        final = self._last_result["scores"]["final"]
        fx = int(final * total_w * 0.97)
        self.decomp_canvas.create_line(
            fx, y0 - 8, fx, y0 + bar_h + 8,
            fill="white", width=2, dash=(4, 2))
        self.decomp_canvas.create_text(
            fx, y0 - 12,
            text=f"Final {final:.2f}",
            font=("Segoe UI", 8), fill="white")

        for w in self.token_frame.winfo_children():
            w.destroy()

        tokens = xai_output["tokens"]
        if tokens:
            max_imp = max(abs(t["importance"]) for t in tokens) or 1.0

            wrap_frame = tk.Frame(self.token_frame, bg="#f9f9f9")
            wrap_frame.pack(fill="x", padx=6, pady=4)

            for tok in tokens:
                imp = tok["importance"]
                intensity = min(int(abs(imp) / max_imp * 200), 200)

                if tok["direction"] == "risk":
                    red = 255
                    green_blue = max(0, 200 - intensity)
                    bg = f"#{red:02x}{green_blue:02x}{green_blue:02x}"
                else:
                    green = 255
                    red_blue = max(0, 200 - intensity)
                    bg = f"#{red_blue:02x}{green:02x}{red_blue:02x}"

                fg = "white" if intensity > 120 else "#1a1a1a"

                lbl = tk.Label(
                    wrap_frame,
                    text=f" {tok['token']} ",
                    font=("Segoe UI", 10),
                    bg=bg, fg=fg,
                    relief="flat", padx=3, pady=2,
                    cursor="hand2")
                lbl.pack(side="left", padx=2, pady=2)

                _imp_str = f"+{imp:.3f}" if imp >= 0 else f"{imp:.3f}"
                lbl.bind("<Enter>", lambda e, l=lbl, s=_imp_str, t=tok["token"]:
                         self.status_label.config(
                             text=f"'{t}'  attribution: {s}  ({tok['direction']})",
                             fg="#666666"))
                lbl.bind("<Leave>", lambda e:
                         self.status_label.config(
                             text="● Hover over tokens to see individual attributions",
                             fg="#666666"))

        else:
            tk.Label(self.token_frame,
                     text="No tokens to display.",
                     font=("Segoe UI", 9, "italic"),
                     bg="#f9f9f9", fg="#999999").pack(anchor="w", padx=6, pady=4)

        self.xai_text.config(state=tk.NORMAL)
        self.xai_text.delete("1.0", tk.END)
        self.xai_text.insert(tk.END, xai_output["explanation"])
        self.xai_text.config(state=tk.DISABLED)

        self.xai_btn.config(state=tk.NORMAL, text="Explain Decision")
        self.status_label.config(
            text="● XAI complete — hover over coloured tokens for details",
            fg="#4ade80")
        self.root.after(200, lambda: self.canvas.yview_moveto(1.0))


if __name__ == "__main__":
    root = tk.Tk()
    app = VishingApp(root)
    root.mainloop()
