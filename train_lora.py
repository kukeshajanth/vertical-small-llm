"""Fine-tune a small instruct model on a vertical with (Q)LoRA.

Hero: genuine 3B (SmolLM3-3B), so the article's "3B" stays literal.
DGX Spark (128GB): add --no-4bit for a clean bf16 LoRA.
24GB GPU / Colab: leave the default 4-bit on.

Prints train_seconds so you can quote the real number in the article.
"""
import argparse
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from data import REGISTRY, build_prompt, examples, system_prompt


def chat_text(tok, name, ex):
    """Render one training example with the assistant label; thinking off (SmolLM3 etc.)."""
    msgs = [
        {"role": "system", "content": system_prompt(name)},
        {"role": "user", "content": build_prompt(name, ex["text"], with_labels=False)},
        {"role": "assistant", "content": ex["label"]},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    except TypeError:  # tokenizer has no reasoning toggle
        return tok.apply_chat_template(msgs, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(REGISTRY), default="ledgar")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-size", type=int, default=5000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--no-4bit", action="store_true",
                    help="bf16 LoRA instead of 4-bit QLoRA (use on DGX Spark / big VRAM)")
    args = ap.parse_args()

    out = args.out or f"adapters/{args.dataset}-smollm3-3b"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, dtype=torch.bfloat16, device_map="auto"
    )

    rows = [{"text": chat_text(tok, args.dataset, ex)}
            for ex in examples(args.dataset, "train", n=args.train_size)]
    ds = Dataset.from_list(rows)

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear",   # 2026 default; was q/k/v/o + MLP
    )

    cfg = SFTConfig(
        output_dir=out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_length,   # older TRL: rename to max_seq_length
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg,
        processing_class=tok,   # older TRL: tokenizer
    )

    t = time.time()
    trainer.train()
    print(f"train_seconds={time.time() - t:.1f}")

    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"saved adapter -> {out}")


if __name__ == "__main__":
    main()
