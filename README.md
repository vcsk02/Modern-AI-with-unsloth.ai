# Modern AI with Unsloth.ai — Coursework

This repository contains five lightweight, **Kaggle-ready** notebooks that demonstrate modern fine-tuning and RL techniques for small open-weights LLMs using 🤗 Transformers, TRL, PEFT, and Unsloth.

> **Deliverables covered**
> - (a) Full finetuning on a tiny model  
> - (b) LoRA (parameter-efficient finetuning) on the same model  
> - (c) RL from preferences (DPO)  
> - (d) RL for reasoning (GRPO)  
> - (e) Continued pretraining (teach a new “language”)  
> - Each notebook includes clear cells for **dataset format**, **training**, **inference**, and **saved artifacts** for grading.  
> - Designed to run quickly on a **single T4 GPU** in Kaggle.  

---

## Contents

- `unsloth_hw_colab1.ipynb` — **Full finetuning** of `HuggingFaceTB/SmolLM2-135M` with a tiny instruction dataset using 🤗 `Trainer`.  
  - Includes an **optional commented** Unsloth snippet showing where `full_finetune=True` goes for Unsloth-supported bases (e.g., Gemma 1B), for when you have more VRAM.

- `unsloth_hw_colab2.ipynb` — **LoRA (PEFT)** on the same base model.  
  - Targets attention/MLP projection modules; includes optional **adapter merge** for standalone inference.

- `unsloth_hw_colab3.ipynb` — **DPO** (Direct Preference Optimization).  
  - Uses a micro **prompt/chosen/rejected** dataset. Configured for **TRL ≥ 0.23** (uses `processing_class=tokenizer`, `beta` in `DPOConfig`).

- `unsloth_hw_colab4.ipynb` — **GRPO** (Group Relative Policy Optimization) for reasoning.  
  - Tiny arithmetic tasks with a simple exact-match reward on `Answer: <num>`. Configured for **TRL ≥ 0.23**.

- `unsloth_hw_colab5.ipynb` — **Continued pretraining** (CPT) on a small synthetic “Quirkish” corpus.  
  - Causal LM objective (`mlm=False`), short run, sample generation in the new style.

> **Note**: If your repo uses a `notebooks/` folder, move the files there and update links accordingly.

---

## Quick Start (Kaggle)

1. **Create Notebook → Upload** one of the `.ipynb` files from this repo.  
2. In **Settings**:  
   - **Accelerator**: **GPU T4** (single, *not* ×2)  
   - **Internet**: **On**  
3. **Run all cells**. Each notebook:  
   - Builds a tiny dataset (fast)  
   - Trains for a small number of steps  
   - Shows quick **inference**  
   - Saves artifacts under `/kaggle/working/...` (zipped for download)

---

## Recording Guide (3–5 minutes per notebook)

Use the same structure for consistency:

1. Show **Kaggle settings** (GPU T4, Internet On).  
2. Run **install + version check** (print Torch/Transformers/TRL versions, GPU name).  
3. Explain the **dataset format** (for DPO: `prompt/chosen/rejected`, for GRPO: `prompt/answer/answer_only`, for CPT: raw `text`).  
4. **Training**: highlight key arguments (short `max_steps`, evaluation strategy, dtype).  
5. **Metrics**: print eval loss and optional **perplexity** (when applicable).  
6. **Inference**: sample with a prompt that matches the training template.  
7. **Save/Zip** model artifacts and point at the files in the sidebar.

---

## Environments & Models

- Base model for speed: **`HuggingFaceTB/SmolLM2-135M`** (tiny, runs fast on T4).  
- **Precision on T4**: use **fp16**. T4s do **not** support bfloat16 (bf16).  
- Tested libs (Kaggle runtime may vary):
  - **Transformers** ≥ 4.57  
  - **TRL** ≥ 0.23 (new APIs)  
  - **PEFT** ≥ 0.12  
  - **Accelerate** ≥ 0.33  
  - **Torch** 2.6+  

If pip prints resolver warnings about unrelated Kaggle packages (RAPIDS/BigQuery), they’re safe to ignore for these notebooks.

---

## Data Formats

- **Supervised FT / LoRA**  
  - Minimal pair: `{ "instruction": "...", "response": "..." }`  
  - Mapped to a single `text` with a chat/instruction template (documented in the notebook).

- **DPO (Preferences)**  
  - `{ "prompt": "...", "chosen": "...", "rejected": "..." }`  
  - TRL ≥ 0.23: pass `processing_class=tokenizer` to the trainer and lengths via `DPOConfig`.

- **GRPO (Reasoning)**  
  - `{ "prompt": "...", "answer": "chain of thought ... Answer: 52", "answer_only": "Answer: 52" }`  
  - Reward: +1 if completion contains `answer_only`, else 0.

- **CPT (Continued Pretraining)**  
  - Raw text lines: `{ "text": "..." }`  
  - Causal LM objective (`mlm=False`).

Swap the tiny toy data for your real datasets when you need stronger results.

---

## Troubleshooting (common, quick fixes)

- **`TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`**  
  - Newer Transformers use `eval_strategy="steps"` and `save_strategy="steps"`. Update arg names.

- **DPO (TRL ≥ 0.23)**  
  - Use `processing_class=tokenizer` (not `tokenizer=`).  
  - Put `beta`, lengths (`max_prompt_length`, `max_completion_length`, `max_length`) in **`DPOConfig`**, not the trainer call.

- **GRPO (TRL ≥ 0.23)**  
  - Ensure `generation_batch_size` is **divisible** by `num_generations`. For quick demos: set both to **1**.  
  - Put lengths and generation knobs in **`GRPOConfig`**.

- **T4 + bf16 error**  
  - Load models in **float16**, set `fp16=True, bf16=False` in TRL configs.  
  - Error example: `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'`.

- **Multi-GPU T4×2 slowing down**  
  - Prefer a **single T4** for short runs. Or force one GPU: `os.environ["CUDA_VISIBLE_DEVICES"]="0"` (before imports).

- **Pip resolver warnings**  
  - From Kaggle’s preinstalled RAPIDS/BigQuery stack. Safe to ignore for this project.

---

## (Optional) Export to Ollama

If you export a fine-tuned model to **Ollama**, ensure the **chat template** matches the base family (Llama/Gemma/Qwen). A mismatched template is the most common cause of odd generations. The LoRA notebook includes an **adapter merge** step to produce a standalone model first.

---

## Results (example expectations)

- These notebooks are sized to **run quickly**, not to reach SOTA.  
- You should see:
  - Loss decreasing within ~20–120 steps.  
  - Reasonable completions for the task template.  
  - Saved artifacts under `/kaggle/working/...` (zipped for download/submit).

---

## Project Structure (suggested)

```
Modern-AI-with-unsloth.ai/
├─ notebooks/
│  ├─ unsloth_hw_colab1.ipynb  # Full FT
│  ├─ unsloth_hw_colab2.ipynb  # LoRA
│  ├─ unsloth_hw_colab3.ipynb  # DPO
│  ├─ unsloth_hw_colab4.ipynb  # GRPO
│  └─ unsloth_hw_colab5.ipynb  # Continued pretraining
└─ README.md
```

---

## Acknowledgments

- **Unsloth.ai** documentation & tutorials  
- **Hugging Face** Transformers, TRL, PEFT, Datasets, Accelerate  
- **Kaggle** for free GPU notebooks

---

## License

Add a LICENSE file (MIT recommended). If none is present, content is provided “as-is” for educational use.
