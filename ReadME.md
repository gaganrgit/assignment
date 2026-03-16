# Kapture CX — ML Intern Take-Home Assignment
**Submitted by:** UVCE B.Tech Student Gagan R
**Assignment:** Kapture CX 

---

## Repository Structure

```
repo/
├── README.md
├── requirements.txt
├── part_a/
│   ├── generate_dataset.py          # Script to regenerate raw dataset
│   ├── raw_conversations.jsonl      # Generated messy dataset (100 convs)
│   ├── clean_data.py                # Data cleaning pipeline
│   ├── quality_report.py            # Quality analysis script
│   ├── cleaned_conversations.jsonl  # Output: clean data (90 convs)
│   ├── rejected_conversations.jsonl # Output: rejected with reasons (10 convs)
│   └── writeup.md                   # Assumptions & reflections
└── part_b/
    ├── finetune.ipynb               # Colab notebook (runnable on T4)
    ├── eval.py                      # Evaluation scorecard
    └── finetune_writeup.md          # Model choice, LoRA reasoning, reflections
```

---

## Part A — Setup & How to Run

### Requirements
```bash
pip install -r requirements.txt
```

### Step 1 — (Optional) Re-generate the raw dataset
```bash
cd part_a
python generate_dataset.py
```
This regenerates `raw_conversations.jsonl` with the same seed. The file is already included.

### Step 2 — Run the cleaning pipeline
```bash
cd part_a
python clean_data.py
```
Outputs:
- `cleaned_conversations.jsonl` — 90 valid conversations
- `rejected_conversations.jsonl` — 10 rejected conversations with `rejection_reason`

### Step 3 — Generate the quality report
```bash
cd part_a
python quality_report.py
```

### Injected Quality Issues (documented)
| Issue Type | Conversation IDs |
|---|---|
| Empty/whitespace turns | conv_011, conv_022, conv_033, conv_044, conv_055 |
| Duplicate consecutive turns | conv_012, conv_023, conv_034, conv_045, conv_056 |
| Fewer than 2 turns | conv_013, conv_024, conv_035 |
| Invalid metadata | conv_014, conv_025, conv_036, conv_047, conv_058 |
| Language label mismatch | conv_015, conv_026, conv_037, conv_048, conv_059 |
| Garbled encoding | conv_016, conv_027, conv_038, conv_049, conv_060 |

---

## Part B — Finetuning on Google Colab

### Colab Instructions
1. Open `part_b/finetune.ipynb` in Google Colab
2. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4
3. Upload `part_a/cleaned_conversations.jsonl` when prompted
4. Run all cells in order (Runtime → Run all)

The notebook will:
- Install all dependencies
- Load `Qwen2.5-0.5B-Instruct` with 4-bit quantisation
- Format training data using the model's chat template
- Apply LoRA (rank=16, alpha=32, target: q_proj + v_proj)
- Train for 3 epochs
- Save adapter weights to `./emi_agent_adapter/`
- Print before/after inference comparison on 5 prompts
- Print a 10-prompt evaluation scorecard

### Running eval.py Standalone (no GPU needed)
```bash
cd part_b
python eval.py
```
Without a GPU/adapter, it scores pre-written sample responses to demonstrate the scoring logic.

### LoRA Adapter Weights
Adapter weights are saved by the notebook to `./emi_agent_adapter/` inside Colab.  
Download via: Files panel → right-click `emi_agent_adapter` → Download.  
To reproduce: re-run the notebook end-to-end (deterministic with `do_sample=False`).

---

## Resources & Citations
- Qwen2.5 model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- HuggingFace PEFT library: https://github.com/huggingface/peft
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- LoRA paper: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- BitsAndBytes 4-bit quantisation: https://github.com/TimDettmers/bitsandbytes
- Dataset generation assisted by Claude (Anthropic) for synthetic conversation creation
