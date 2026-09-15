# Task 3 (Advanced) — AI-Driven NLP: Language Model Deployment & Analysis ✅

## 📌 Problem Statement
Deploy a Language Model of choice, implement it from scratch in a Jupyter notebook, and conduct an
in-depth analysis of its performance and capabilities: research questions, strengths/limitations,
alignment with NLP/ML goals, ethical considerations, and conclusions.

## 🧠 Chosen Language Models
| Model | Type | Task in this project | Why |
|---|---|---|---|
| **DistilBERT** (SST-2 fine-tuned) | Encoder Transformer | Sentiment analysis | BERT's strengths at 60% size — free, fast on CPU |
| **DistilGPT-2** | Decoder Transformer | Text generation | GPT-2 lineage — coherent, lightweight, reproducible |

Both run **locally on CPU via Hugging Face** — no API keys, fully reproducible, same transformer
methodology as the giant models (GPT-3/BERT) named in the task brief.

## 🔬 Research Questions & Measured Findings
1. **RQ1 — Negation:** 4/4 sentence pairs flipped sentiment correctly when negated (SST-2 fine-tuning works),
   with lower confidence on subtle constructions ("hardly", "anything but").
2. **RQ2 — Calibration:** average confidence 0.92 on ambiguous/sarcastic text vs 0.96 on clear text —
   the model is **overconfident out-of-distribution** → production fix: confidence threshold → human review.
3. **RQ3 — Temperature:** at 0.3 the generator **collapsed into newline loops** (line-repetition 0.946,
   13 real words); at 1.3 grammar degraded; 0.8 / top-p 0.95 = sweet spot. Multi-metric design
   (line-repetition + distinct-2 + content length) was needed to catch the collapse.

**Fairness probe:** identity-swap pairs (names/pronouns) → identical labels, max Δconfidence = 0.0000,
with documented caveats.

## 📝 Notebook Contents
LM selection rationale · environment versions · tokenization internals (tokens/IDs/attention masks)
· labeled benchmark & confidence visualization · negation experiment · calibration experiment ·
bias probe · generation parameters (temperature/top-k/top-p) with measured metrics · ecosystem map
(OpenAI vs Hugging Face vs **LangChain**, matching the internship brief diagram) · ethics · conclusions ·
reproduction notes.

## ⚖️ Ethical Considerations
Bias audits, hallucination grounding (RAG), misuse filtering, on-premise privacy via local models,
smaller distills as an energy choice, AI-content transparency — discussed with mitigations in §11.

## ▶️ How to Run
```bash
pip install transformers torch pandas matplotlib
jupyter notebook Language_Model_NLP_Project.ipynb
# models (~600MB) auto-download once from Hugging Face, then cached
```

📖 Read **`EXPLAIN.md`** — the 60-second story, the numbers to quote, and 10 Q&A model answers.
