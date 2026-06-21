#!/usr/bin/env python3
"""
LoRA fine-tuning of Qwen2.5-0.5B on episode-prefix training pairs (ADR 0033).

Usage:
    python3 scripts/train_lora.py \
        --training-data memory/lora_training_data.jsonl \
        --proj-weights-dir memory/proj_weights \
        --model-id Qwen/Qwen2.5-0.5B \
        --output-dir memory/lora_output \
        [--epochs 3] [--lr 2e-4] [--rank 8] [--alpha 16] [--batch-size 4]

After training, merge adapters and export:
    python3 scripts/train_lora.py ... --export-onnx models/qwen25_lm.onnx

Prerequisites:
    pip install peft transformers torch accelerate onnx

The script:
1. Loads (episode, text) pairs from the JSONL training file.
2. Loads MLP projection weights (w1.bin, w2.bin) exported via --export-proj-weights.
3. Applies LoRA adapters to Qwen2.5-0.5B q_proj and v_proj layers.
4. For each pair: project episode -> 5 region tokens, prepend to text, teacher-force.
5. Merges adapters and optionally exports to ONNX.
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: peft and transformers required. Run:")
    print("  pip install peft transformers torch accelerate")
    sys.exit(1)

# ── constants matching action.rs ─────────────────────────────────────────────
EPISODE_DIM = 640
N_REGIONS = 5
LM_HIDDEN = 896
PROJ_HIDDEN = 1024


def load_proj_weights(proj_dir: str, linear_proj: bool = False):
    """Load MLP projection weights from w1.bin and w2.bin."""
    w1_path = os.path.join(proj_dir, "w1.bin")
    w2_path = os.path.join(proj_dir, "w2.bin")

    w1_flat = np.frombuffer(open(w1_path, "rb").read(), dtype=np.float32)
    if linear_proj:
        # Single linear: [EPISODE_DIM, N_REGIONS * LM_HIDDEN]
        w1 = w1_flat.reshape(EPISODE_DIM, N_REGIONS * LM_HIDDEN)
        w2 = None
    else:
        # MLP: W1=[EPISODE_DIM, PROJ_HIDDEN], W2=[PROJ_HIDDEN, N_REGIONS*LM_HIDDEN]
        w1 = w1_flat.reshape(EPISODE_DIM, PROJ_HIDDEN)
        w2_flat = np.frombuffer(open(w2_path, "rb").read(), dtype=np.float32)
        w2 = w2_flat.reshape(PROJ_HIDDEN, N_REGIONS * LM_HIDDEN)

    return w1, w2


def project_episode(episode: np.ndarray, w1: np.ndarray, w2) -> np.ndarray:
    """Project 640-dim episode vector -> [N_REGIONS, LM_HIDDEN] region tokens."""
    if w2 is None:
        # Linear projection
        out = episode @ w1  # [N_REGIONS * LM_HIDDEN]
    else:
        # MLP: relu(x @ W1) @ W2
        h = np.maximum(0.0, episode @ w1)  # [PROJ_HIDDEN]
        out = h @ w2  # [N_REGIONS * LM_HIDDEN]
    return out.reshape(N_REGIONS, LM_HIDDEN).astype(np.float32)


class EpisodeTextDataset(Dataset):
    """(episode_tokens, text_token_ids) pairs for LoRA training."""

    def __init__(self, jsonl_path: str, tokenizer, w1, w2, max_text_len: int = 128):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                episode = np.array(item["episode"], dtype=np.float32)
                text = item["text"].strip()
                if not text:
                    continue
                region_tokens = project_episode(episode, w1, w2)  # [5, 896]
                text_ids = tokenizer.encode(
                    text, add_special_tokens=False, truncation=True, max_length=max_text_len
                )
                if len(text_ids) < 2:
                    continue
                self.samples.append((region_tokens, text_ids))

        print(f"Loaded {len(self.samples)} training pairs from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Collate a list of (region_tokens, text_ids) into padded tensors."""
    region_tokens = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    max_len = max(len(b[1]) for b in batch)
    text_ids_padded = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    for i, (_, ids) in enumerate(batch):
        text_ids_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return region_tokens, text_ids_padded


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model.
    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32, trust_remote_code=True
    )

    # Apply LoRA.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)

    # Load projection weights.
    w1, w2 = load_proj_weights(args.proj_weights_dir, linear_proj=args.linear_proj)
    print(f"Projection weights loaded from {args.proj_weights_dir}")

    # Dataset.
    dataset = EpisodeTextDataset(
        args.training_data, tokenizer, w1, w2, max_text_len=args.max_text_len
    )
    if len(dataset) == 0:
        print("ERROR: no training pairs found. Run a campaign with --export-training-data first.")
        sys.exit(1)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(dataset) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Frozen embedding table for prefix construction.
    # Qwen2.5-0.5B ties lm_head and embed_tokens; grab the embedding matrix.
    embed_weight = model.base_model.model.model.embed_tokens.weight  # [vocab, 896]

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_n = 0
        for region_tokens, text_ids in loader:
            # region_tokens: [B, 5, 896], text_ids: [B, T] (padded with -100)
            region_tokens = region_tokens.to(device)
            text_ids = text_ids.to(device)

            # Build input embeds: [prefix_region_tokens | text_token_embeds]
            # region_tokens already in embedding space (896-dim), reshape to [B, 5, 896].
            # text_ids: replace -100 padding with 0 for embedding lookup (labels handle loss mask).
            valid_ids = text_ids.clone()
            valid_ids[valid_ids == -100] = 0
            text_embeds = embed_weight[valid_ids]  # [B, T, 896]

            # Concatenate: [B, 5 + T, 896]
            inputs_embeds = torch.cat([region_tokens, text_embeds], dim=1)

            # Labels: -100 for region prefix positions, then the text_ids shifted left by 1
            # (standard causal LM: predict next token at each position).
            # We want to predict text_ids[1:] given text_ids[:-1] as context.
            # With the prefix, positions 0..4 are region tokens (label -100),
            # positions 5..5+T-1 are text embeddings; labels at pos 5+i = text_ids[i+1].
            B, T = text_ids.shape
            prefix_ignore = torch.full((B, N_REGIONS), -100, dtype=torch.long, device=device)
            # Shift: predict text_ids at positions 1..T given prefix+text[0..T-1]
            # Label for position 5+i is text_ids[i+1]; last text position has no next token.
            text_labels_shifted = torch.cat(
                [text_ids[:, 1:], torch.full((B, 1), -100, dtype=torch.long, device=device)], dim=1
            )
            labels = torch.cat([prefix_ignore, text_labels_shifted], dim=1)  # [B, 5+T]

            outputs = model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_n += 1
            global_step += 1

            if global_step % 50 == 0:
                print(f"  step {global_step}/{total_steps}  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        print(f"Epoch {epoch}/{args.epochs}  avg_loss={epoch_loss/max(epoch_n,1):.4f}")

    # Save LoRA adapters.
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapters saved to {args.output_dir}")

    # Merge and export ONNX if requested.
    if args.export_onnx:
        print("Merging LoRA adapters ...")
        merged = model.merge_and_unload()
        merged.eval()
        export_onnx(merged, tokenizer, args.export_onnx, device)


def export_onnx(model, tokenizer, onnx_path: str, device):
    """Export the merged model to ONNX (dynamic seq_len, inputs_embeds interface)."""
    try:
        import onnx
    except ImportError:
        print("WARNING: onnx not installed — skipping ONNX export. Run: pip install onnx")
        return

    print(f"Exporting ONNX to {onnx_path} ...")

    # Back up existing model.
    if os.path.exists(onnx_path):
        backup = onnx_path.replace(".onnx", "_pretrained.onnx")
        os.rename(onnx_path, backup)
        print(f"  backed up existing model to {backup}")

    # Dummy input: [1, 5, 896] region token embeddings.
    dummy_embeds = torch.zeros(1, N_REGIONS, LM_HIDDEN, dtype=torch.float32, device=device)

    torch.onnx.export(
        model,
        {"inputs_embeds": dummy_embeds},
        onnx_path,
        input_names=["inputs_embeds"],
        output_names=["last_hidden_state"],
        dynamic_axes={"inputs_embeds": {0: "batch", 1: "seq"}, "last_hidden_state": {0: "batch", 1: "seq"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX export complete: {onnx_path}")
    print("NOTE: run scripts/export_models.py to rebuild qwen25_gain.onnx and qwen25_text_hidden.onnx")
    print("      if gain drift is observed after LoRA deployment (see ADR 0033).")


def main():
    p = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-0.5B on episode prefixes")
    p.add_argument("--training-data", required=True, help="JSONL file from --export-training-data")
    p.add_argument("--proj-weights-dir", required=True, help="Dir with w1.bin / w2.bin from --export-proj-weights")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B", help="HuggingFace model id or local path")
    p.add_argument("--output-dir", default="memory/lora_output", help="Where to save LoRA adapter weights")
    p.add_argument("--export-onnx", default=None, help="Path to write merged ONNX (e.g. models/qwen25_lm.onnx)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-text-len", type=int, default=128, help="Max tokens per text target")
    p.add_argument("--linear-proj", action="store_true", help="Use linear projection (no hidden layer)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
