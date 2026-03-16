# Part B — Finetuning Writeup

## Model Choice: Qwen2.5-0.5B-Instruct

I chose `Qwen2.5-0.5B-Instruct` for the following reasons:

1. **Fits on T4 free tier comfortably.** At 0.5B parameters the model loads in ~1 GB of GPU RAM, leaving plenty of headroom for activations, optimizer states (LoRA-only), and a reasonable batch size.

2. **Instruction-tuned base.** The `-Instruct` variant already follows chat templates and system prompts, which means less work is needed to get structured agent-like responses out of the box. We're not teaching it *how* to follow instructions — just *what persona* to adopt.

3. **Multilingual pretraining.** Qwen models are pretrained on a large multilingual corpus including Hindi and other South Asian languages. This gives it a reasonable prior for Hinglish even before finetuning, unlike models trained almost exclusively on English.

4. **Active HuggingFace ecosystem support.** Up-to-date tokenizer, chat template, and PEFT compatibility are all confirmed.

---

## LoRA Configuration and Reasoning

| Hyperparameter | Value | Reasoning |
|---|---|---|
| `r` (rank) | 16 | Low rank captures task-specific directions without overfitting on 50–100 examples. Rank 8 is often cited as sufficient; 16 gives a bit more capacity for Hinglish style shift. |
| `lora_alpha` | 32 | Standard 2× rank ratio. Scales the LoRA update so the effective learning rate isn't too small. |
| `lora_dropout` | 0.05 | Light regularisation to prevent overfitting on the small dataset. |
| `target_modules` | `q_proj`, `v_proj` | Attention query and value projections are the most impactful modules for style and persona adaptation. Adding `k_proj` and `o_proj` improves quality but increases parameters; kept minimal for Colab. |
| `task_type` | `CAUSAL_LM` | Standard for autoregressive instruction tuning. |

The total number of trainable parameters is roughly **800K** out of 500M — about 0.16% of the model. This is intentional: LoRA preserves the model's general language capabilities while nudging it toward the EMI agent persona.

---

## What Went Well

- The end-to-end pipeline (load → format → LoRA config → train → save → infer) works cleanly on Colab T4 with no OOM errors.
- The model's responses post-finetuning are more grounded in EMI/payment vocabulary than the base model's generic responses.
- The language detection heuristic in `eval.py` correctly identifies Hinglish responses even without a dedicated language model.

## What Didn't Go Well

- **50–90 training examples is far too few** for visible quality improvement. The before/after difference is subtle. The model sometimes reverts to English or generic assistant tone.
- The **chat template formatting is model-specific** and easy to get wrong. Qwen2.5 uses `<|im_start|>` / `<|im_end|>` tokens; using the wrong template silently produces bad training targets.
- Training for 1 epoch means the model sees each example once — barely enough to register the distribution shift.

## What I'd Do Differently with More Time and Compute

1. **More and better data.** Use the Claude/Gemini API to generate 500–1,000 high-quality Hinglish EMI conversations with varied customer personas (angry, cooperative, financially stressed). Data quality matters more than LoRA hyperparameters at this scale.

2. **Full instruction dataset format review.** Apply the model's tokenizer and inspect decoded training samples visually before training, to catch template errors early.

3. **Learning rate sweep.** The default `2e-4` LR works but may not be optimal. A quick sweep over `[1e-4, 2e-4, 5e-4]` with 3 epochs each would clarify.

---

## Top 3 Priorities to Improve Quality for Production

1. **Data scale and diversity.** Increase to 2,000+ training conversations covering edge cases: irate customers, partial payments, dispute escalations, callbacks. Include real transcripts (anonymised) if available. No LoRA config change will substitute for better training data.

2. **Evaluation with real metrics.** Replace the keyword pass/fail scorecard with a proper evaluation: BLEU/ROUGE against reference responses, a secondary LLM-as-judge (e.g., GPT-4o) rating politeness + relevance, and human spot-checks. You can't improve what you don't measure.

3. **Merge and quantise for deployment.** Merge the LoRA adapter into the base weights (`peft.merge_and_unload()`), then apply 4-bit GPTQ or AWQ quantisation. This reduces serving latency and memory footprint by 3–4×, making real-time voice agent inference feasible on modest hardware.
