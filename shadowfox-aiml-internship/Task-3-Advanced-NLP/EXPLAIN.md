# 🎓 Task 3 — How to Explain Your Project (Deploying & Analyzing Language Models)

## The 60-second story (memorize this)

> "For the advanced task I deployed and analyzed two open Language Models locally:
> **DistilBERT** (an encoder, fine-tuned for sentiment) and **DistilGPT-2** (a decoder, for
> generation) — the same family of technology as BERT/GPT-3, small enough to run free on CPU.
> I didn't just call the API — I opened the hood: tokenization, IDs, attention masks — then
> ran three measured experiments. **RQ1 negation:** DistilBERT flipped sentiment correctly on
> 4/4 negation pairs, but with lower confidence on subtle negation. **RQ2 calibration:** on
> ambiguous and sarcastic text the model stayed 0.92 confident on average — nearly as confident
> as on clear text (0.96) — it's *overconfident*, so in production you'd add a confidence
> threshold routing to humans. **RQ3 temperature:** measured with three metrics —
> at 0.3 the generator collapsed into newline loops (line-repetition 0.95), at 1.3 grammar
> degraded, 0.8 was the sweet spot. I also ran a fairness probe — identical sentences with
> different names gave identical sentiment (Δconfidence 0.0000) — and documented ethics:
> bias, hallucination, misuse. Finally I mapped where this sits in the industry stack:
> Hugging Face models under a LangChain application layer, the exact diagram in the task PDF."

## The two-model idea in one table

| | DistilBERT | DistilGPT-2 |
|---|---|---|
| Transformer type | **Encoder** — reads whole sentence at once | **Decoder** — predicts next token left→right |
| Job | understanding (sentiment here) | open-ended generation |
| Trained by | masked-token guessing + SST-2 fine-tune | next-token prediction on web text |
| Params | 66M | 82M |
| In the notebook | `pipeline("sentiment-analysis")` | `pipeline("text-generation")` |

## Concept cheat-sheet

| Concept | Say it like this |
|---|---|
| **Token** | "The LM's atom of text — often a word-piece: 'unbelievable' → `un`, `##believ`, `##able`. We printed real tokens in §3." |
| **[CLS] / [SEP]** | "Special wrapper tokens; [CLS] collects a sentence-level summary that the sentiment head reads." |
| **Attention mask** | "1 for real tokens, 0 for padding — tells the transformer what to ignore." |
| **Self-attention** | "Every token re-reads every other token to understand context — why 'bank' means river or money depending on the sentence." |
| **pipeline()** | "Hugging Face's one-liner that bundles tokenizer + model + post-processing into a callable." |
| **Fine-tuning** | "Taking a general pre-trained LM and training its last layers on a specific task (SST-2 reviews → sentiment)." |
| **Encoder vs decoder** | "BERT understands, GPT writes. That's why GPT-3 in the task brief is a decoder — and I analyzed one of each." |
| **Distillation** | "A small 'student' model trained to mimic a big 'teacher' — DistilBERT keeps ~97% of BERT's skill at 60% of the speed." |
| **Temperature** | "Rescales next-token randomness; <1 conservative, >1 chaotic. We *measured* its effect instead of quoting folklore." |
| **top-k / top-p** | "Sampling pools: only the top 50 tokens / the smallest 95%-probability nucleus can be drawn. Cuts the wild tail." |
| **line-repetition metric** | "1 − unique/total lines: 0.95 at temp 0.3 proved *repetition collapse* (the 'Curious Case of Neural Text Degeneration' paper)." |
| **distinct-2** | "Unique bigrams ÷ total bigrams — lexical richness of the *cleaned* content." |
| **Calibration** | "Does '0.99 confident' mean 99% correct? Our RQ2 says no on ambiguous text — a known classifier weakness (OOD)." |
| **Hallucination** | "Generators state false things fluently — headline risk for any real product." |

## Your measured results (quote these numbers)

| Experiment | Result | One-line takeaway |
|---|---|---|
| Clear benchmark (10 sentences, mixed domains incl. cricket & monsoon) | **10/10 = 100%** | accurate on clear sentiment |
| RQ1 negation pairs | **4/4 labels flipped** | SST-2 fine-tuning taught negation; subtle cases lower confidence |
| RQ2 ambiguous/sarcastic | avg confidence **0.92** (vs 0.96 clear) | overconfident — needs confidence thresholding to humans |
| Fairness probe (4 name/pronoun pairs) | **Δconfidence = 0.0000**, no label disagreement | no surface bias on this probe — larger audits still needed |
| RQ3 temp 0.3 / 0.8 / 1.3 | line-repetition **0.946 / 0.466 / 0.300**; content words 13 / 34 / 37.5 | low temp = loop collapse, high temp = grammar slips, 0.8 sweet spot |

## Likely mentor questions → your answers

1. **Why these models instead of GPT-3?** — "Same transformer methodology, but open weights: free, offline, private, reproducible — and the analysis transfers 1:1."
2. **What is the transformer in one sentence?** — "A stack of self-attention layers that let every token weigh every other token to build contextual meaning."
3. **Why did low temperature fail?** — "Sharpening the distribution makes the most probable token win repeatedly — the model falls into loops; measured line-repetition 0.95."
4. **What is top-p (nucleus sampling)?** — "Keep the smallest token set covering 95% probability mass; adapts pool size to the model's certainty, unlike fixed top-k."
5. **How would you improve sentiment on sarcasm?** — "A third class (neutral), confidence threshold → human review, and fine-tuning on domain data with sarcasm labels."
6. **What hallucination risk does DistilGPT-2 pose?** — "It states plausible-sounding falsehoods; never serve generations unverified — retrieval grounding (RAG) is the standard fix."
7. **How would LangChain fit in?** — "As the app layer from the PDF diagram: wrap my HF pipeline in a few lines, then compose prompt templates, chains, memory, or a RAG flow over documents."
8. **What bias probes did you run?** — "Identity-swap pairs: identical content, different names/pronouns — sentiment was identical (Δ 0.0000). Honest caveat: generation models show larger stereotype effects; audits scale with stakes."
9. **How would you evaluate generation more rigorously?** — "Held-out perplexity, human preference ratings, and task metrics (e.g., ROUGE for summarization); distinct-n is a cheap first signal."
10. **Ethical headline of your project?** — "Models are deployable only with three guards: bias audits, hallucination grounding, and misuse filtering — plus transparency that text is AI-generated."

## What to show in the video

1. §1 table — "two LMs: one understands, one writes"
2. §3 tokenization cell — actual tokens / IDs / attention mask printed
3. §5 negation table — the flipped? column = True ×4
4. §6 confidence bar chart — "red bars are wrong answers with high confidence"
5. §9 raw outputs — the **`\n\n\n…` repetition collapse at temp 0.3** + the 3-bar summary chart
6. §10 stack diagram — "this is the exact LangChain layer from the internship brief"
7. §11–12 ethics + conclusions
