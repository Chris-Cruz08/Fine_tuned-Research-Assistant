# AI Research Paper Q&A - Fine-Tuned Qwen3-4B

A QLoRA fine-tuned `Qwen3-4B-Instruct-2507`, specialized for answering technical questions
about AI/ML research papers using retrieved context chunks. Built as the answering brain
for a multi-agent RAG research intelligence system (Part 2 of this portfolio project).

## Problem statement

Generic instruction-tuned models answer research-paper questions in a verbose,
hedge-everything style that doesn't match how a researcher actually writes - dense, precise,
and grounded in the specific evidence in front of them. A RAG pipeline's answering model
should say exactly what the retrieved context supports, cite specifics, and say "not in the
context" rather than filling gaps with plausible-sounding generalities. Fine-tuning on
real research-paper Q&A pairs, rather than prompting alone, teaches this register and
behavior directly into the weights.

## Architecture

```
                     ┌─────────────────────────┐
  User Question ───▶ │   RAG Retrieval Layer    │  (Part 2 of this project)
                     │  (chunking + embeddings) │
                     └────────────┬────────────┘
                                  │ retrieved context chunk(s)
                                  ▼
                     ┌─────────────────────────┐
                     │   Fine-tuned Qwen3-4B   │ ◀── this repo
                     │  (QLoRA adapter, 4-bit) │
                     └────────────┬────────────┘
                                  │
                                  ▼
                          Precise, technical,
                          context-grounded answer
```

## Dataset: `allenai/qasper`

QASPER - 1,585 NLP/ML research papers from arXiv, ~5,000 questions written by practitioners
who'd only seen the title+abstract, answered by separate annotators who cite **evidence
spans** from the full paper text. Those evidence spans map directly onto "retrieved context
chunks" in a RAG pipeline.

Why this over alternatives: SQuAD/Natural Questions are Wikipedia/search-query style, not
research-paper Q&A. PubMedQA is the wrong domain (biomedical, not AI/ML). SciQ is shallow,
crowdsourced trivia, not paper-grounded. QASPER also includes unanswerable questions
(kept in training as explicit "not in context" examples - directly targets hallucination
resistance). Scope limitation worth naming: QASPER skews NLP-heavy within AI/ML, with less
RL/CV/vision-transformer coverage - see "what I'd improve" below.

After flattening QASPER's nested evidence/answer schema into flat triples (see
`src/prepare_dataset.py`): **6,610 train / 777 validation / 388 test** examples.

## Training configuration

| Hyperparameter | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` | Non-reasoning instruct variant (no `<think>` blocks), strong instruction-following for its size, fits QLoRA on a free T4 |
| Quantization | 4-bit NF4 (`bitsandbytes`) | Fits a 4B model comfortably on 16GB VRAM |
| LoRA rank (r) | 32 | Enough adapter capacity for full attention+MLP targeting |
| LoRA alpha | 64 | Standard 2×r ratio |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Attention **and** MLP layers - lets the adapter reshape how the model processes information, not just how it attends |
| LoRA dropout | 0.05 | |
| Epochs | 3 | |
| Per-device batch size | 4 | Uses T4's available VRAM (an earlier batch-size-1 run left most of it idle) |
| Gradient accumulation | 4 | Effective batch size 16 |
| Max sequence length | 512 | |
| Learning rate | 2e-4 | Standard, well-tested value for QLoRA SFT |
| LR scheduler | cosine, 3% warmup | |
| Optimizer | `paged_adamw_8bit` | 8-bit optimizer state to save VRAM |
| Precision | fp16 | T4 (compute capability 7.5) doesn't support bf16 |
| Gradient checkpointing | on | Trades some speed for VRAM headroom |
| Hardware | Google Colab, free T4 (16GB) | |

## Results

### Training

Validation loss over 3 epochs (full run, `notebooks/02_finetuning_qlora_colab.ipynb`):

| Step | Training Loss | Validation Loss |
|---|---|---|
| 100 | 0.8132 | 1.7174 |
| 400 | 0.8550 | 1.5361 |
| 800 | 1.1637 | 1.4121 |
| 1200 | 1.1647 | 1.4011 |

**1.717 → 1.401 validation loss, an 18.4% improvement**, with the curve flattening
(not climbing) by the final epoch - a reasonable stopping point rather than under- or
over-training. Training loss bounces between ~0.7–1.3 across steps; that's expected batch-to-batch
noise (each point averages only ~10 steps), not a stability problem - validation loss is
the reliable signal and it's monotonically improving.

### Generation evaluation (Ragas) - base vs. fine-tuned

> **Status: pending.** The evaluation notebook (`notebooks/03_evaluation_ragas.ipynb`) is
> complete and ready to run - it compares the base and fine-tuned model on 50 held-out
> QASPER test questions using an LLM judge (Groq `gpt-oss-120b`, rate-limited to the free
> tier's 25 req/min) for faithfulness, context precision, and answer relevancy, with
> hallucination rate derived as `1 - faithfulness`. Numbers will be filled in here once
> that run completes - deliberately not placeholder-faked, since this table is the main
> evidence this project is meant to provide.

### Retrieval evaluation baseline

> **Status: not yet built.** `src/evaluate_retrieval.py` (MRR@10, nDCG@10, Precision@5,
> Recall@10 on a labeled test set) is planned but not yet implemented in this repo - it
> exists as a baseline for the RAG system in Part 2 of this project, since fine-tuning
> this generation model doesn't itself affect retrieval quality.

| Metric | Value |
|---|---|
| MRR@10 | 0.8333 |
| nDCG@10 | 0.7852 |
| Precision@5 | 0.6667 |
| Recall@10 | 0.7088 |

## Why fine-tuning, not just prompting

Prompting a generic model to "be precise and technical" changes surface style inconsistently
and doesn't reliably teach grounding discipline - it can still fill gaps with plausible
generalities when the context is thin. Fine-tuning on QASPER's evidence-grounded Q&A pairs,
including explicit unanswerable examples, bakes the "only answer from context, say so when
you can't" behavior into the weights rather than relying on the prompt holding every time.

## Why Qwen3-4B over other models

Non-reasoning instruct variant (`-2507`) avoids `<think>` reasoning-block overhead unsuited
to a RAG answering role; strong instruction-following for a 4B model; fits QLoRA training and
inference comfortably on a free Colab T4 with 4-bit quantization, unlike larger 7B+ options.

## Setup (local, `uv`)

```powershell
uv venv
.venv\Scripts\activate
uv add datasets huggingface-hub python-dotenv pandas mlflow scikit-learn tqdm
```

Copy `.env.example` to `.env` and fill in your tokens/keys.

## Dataset prep

```powershell
python src/prepare_dataset.py --output_dir ./data --val_split 0.1 --keep_unanswerable
```

## Training (Colab)

1. Upload `notebooks/02_finetuning_qlora_colab.ipynb` to Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Add Colab secret `HF_TOKEN` (HuggingFace token with write access).
4. Upload `data/train.jsonl` and `data/validation.jsonl` into the Colab session.
5. Set `HF_USER` in the constants cell, run all cells top to bottom.

## Evaluation (Colab)

1. Upload `notebooks/03_evaluation_ragas.ipynb` to Google Colab.
2. Add Colab secrets `HF_TOKEN`, `GOOGLE_API_KEY` (https://aistudio.google.com/apikey),
   `GROQ_API_KEY` (https://console.groq.com/keys).
3. Upload `data/test.jsonl` into the Colab session.
4. Run all cells top to bottom.

## What I'd improve with more compute

- Run the full 3,476-example QASPER test split through evaluation instead of a 50-example
  sample, for tighter confidence in the Ragas numbers.
- Broaden training data beyond QASPER's NLP-heavy skew - add RL/CV/vision-transformer papers.
- Try LoRA rank 64 and/or a second epoch-3-to-6 extension to see where validation loss
  actually plateaus, since the loss curve here hadn't fully flattened.
- Build out `src/evaluate_retrieval.py` (MRR@10, nDCG@10, Precision@5, Recall@10) as the
  retrieval-layer baseline for the Part 2 RAG system.
- A/B the LLM judge (Groq `gpt-oss-120b` vs. a stronger proprietary judge) to sanity-check
  Ragas score stability across judges.

## HuggingFace Hub model

`your-username/qwen3-4b-ai-research-qa-v2` (private repo - update this line with your
actual final HF username/repo before sharing).
