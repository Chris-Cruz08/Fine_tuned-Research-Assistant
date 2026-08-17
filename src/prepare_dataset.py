"""
prepare_dataset.py

Flattens the nested `allenai/qasper` schema into flat
(context, question, answer) triples, then formats each triple into the
exact Qwen2.5 chat template the RAG pipeline will use at inference time:

<|im_start|>system
You are an expert AI research assistant. Answer the question based only on
the provided context. Be precise, technical, and cite specific details from
the context.
<|im_end|>
<|im_start|>user
Context: {retrieved_chunk}
Question: {question}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>

Run locally (uv) or in Colab:
    python src/prepare_dataset.py --output_dir ./data --val_split 0.1

Notes on QASPER's raw structure (per paper):
  paper["qas"]["question"][i]      -> question text
  paper["qas"]["answers"][i]["answer"] -> list of answer dicts, each with:
      - "unanswerable": bool
      - "extractive_spans": list[str]
      - "free_form_answer": str
      - "yes_no": bool | None
      - "evidence": list[str]   <- this is our "context chunk"

Multiple annotators can answer the same question; we keep each
(question, evidence, answer) combination as its own training example, and
we DROP unanswerable questions with empty evidence (nothing to ground a
"the context doesn't contain this" example on would just teach a lazy
"I don't know" bias — see --keep_unanswerable if you want them in for
hallucination-resistance training instead).
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are an expert AI research assistant. Answer the question based "
    "only on the provided context. Be precise, technical, and cite "
    "specific details from the context."
)


def build_answer_text(answer: dict) -> str | None:
    """Pick the best textual representation of a QASPER answer object."""
    if answer.get("unanswerable"):
        return None  # handled separately via --keep_unanswerable
    if answer.get("free_form_answer"):
        return answer["free_form_answer"].strip()
    if answer.get("extractive_spans"):
        spans = answer["extractive_spans"]
        return " ".join(s.strip() for s in spans if s.strip())
    if answer.get("yes_no") is not None:
        return "Yes." if answer["yes_no"] else "No."
    return None


def flatten_qasper(raw_split, keep_unanswerable: bool) -> list[dict]:
    rows = []
    for paper in raw_split:
        title = paper.get("title", "")
        qas = paper["qas"]
        for question, answer_group in zip(qas["question"], qas["answers"]):
            for ans_wrapper in answer_group["answer"]:
                evidence = ans_wrapper.get("evidence", [])
                # Filter out figure/table refs with no real text
                evidence = [e.strip() for e in evidence if e and len(e.strip()) > 20]

                if ans_wrapper.get("unanswerable"):
                    if not keep_unanswerable:
                        continue
                    context = " ".join(evidence) if evidence else "(No relevant context found in paper.)"
                    answer_text = "The provided context does not contain information to answer this question."
                else:
                    if not evidence:
                        continue  # no context chunk to ground the answer in -> skip
                    context = " ".join(evidence)
                    answer_text = build_answer_text(ans_wrapper)
                    if not answer_text:
                        continue

                rows.append(
                    {
                        "paper_title": title,
                        "context": context,
                        "question": question.strip(),
                        "answer": answer_text,
                    }
                )
    return rows


def to_qwen_chat_format(row: dict) -> dict:
    user_turn = f"Context: {row['context']}\nQuestion: {row['question']}"
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_turn}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['answer']}<|im_end|>"
    )
    return {
        "text": text,
        "context": row["context"],
        "question": row["question"],
        "answer": row["answer"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--keep_unanswerable", action="store_true",
                         help="Include unanswerable questions as explicit "
                              "'not in context' training examples (recommended "
                              "for lowering hallucination rate).")
    parser.add_argument("--max_context_chars", type=int, default=1800,
                         help="Truncate context to keep tokenized length under "
                              "the 512-token max_seq_length used in training.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading allenai/qasper from HuggingFace Hub...")
    try:
        # Preferred: auto-converted parquet version (works on all recent
        # `datasets` versions, since old-style loading scripts like
        # qasper.py are no longer supported by the library).
        ds = load_dataset("allenai/qasper", revision="refs/convert/parquet")
    except Exception:
        # Fallback for older `datasets` versions that still support scripts.
        ds = load_dataset("allenai/qasper")

    all_rows = []
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            rows = flatten_qasper(ds[split_name], keep_unanswerable=args.keep_unanswerable)
            print(f"  {split_name}: {len(rows)} flattened Q/A pairs")
            all_rows.extend(rows)

    # Truncate overly long contexts so tokenized sequences fit max_seq_length=512
    for row in all_rows:
        if len(row["context"]) > args.max_context_chars:
            row["context"] = row["context"][: args.max_context_chars] + "..."

    random.shuffle(all_rows)
    n = len(all_rows)
    n_test = int(n * args.test_split)
    n_val = int(n * args.val_split)

    test_rows = all_rows[:n_test]
    val_rows = all_rows[n_test:n_test + n_val]
    train_rows = all_rows[n_test + n_val:]

    for split_name, rows in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
        formatted = [to_qwen_chat_format(r) for r in rows]
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in formatted:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(formatted)} examples -> {out_path}")

    print("\nSample formatted example:")
    print(to_qwen_chat_format(train_rows[0])["text"])


if __name__ == "__main__":
    main()
