"""
train.py
----------
python train.py --epochs 10 --batch_size 64

Reads the memmapped arrays produced by tokenization.py — never loads
Data.csv, never loads the full tokenized dataset into RAM. Each batch is
paged in from disk on demand via np.load(..., mmap_mode="r").
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tokenizers import Tokenizer

from model import Transformer


class TranslationDataset(Dataset):
    """Reads pre-tokenized, fixed-length pairs from memmapped .npy files.
    RAM footprint is independent of dataset size."""

    def __init__(self, processed_dir):
        self.src = np.load(f"{processed_dir}/src.npy", mmap_mode="r")
        self.trg = np.load(f"{processed_dir}/trg.npy", mmap_mode="r")
        assert len(self.src) == len(self.trg)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = torch.from_numpy(np.array(self.src[idx], dtype=np.int64))
        trg = torch.from_numpy(np.array(self.trg[idx], dtype=np.int64))
        return src, trg


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tok_en = Tokenizer.from_file(f"{args.tok_dir}/tokenizer_en.json")
    tok_fr = Tokenizer.from_file(f"{args.tok_dir}/tokenizer_fr.json")
    src_pad_idx = tok_en.token_to_id("<pad>")
    trg_pad_idx = tok_fr.token_to_id("<pad>")

    dataset = TranslationDataset(args.processed_dir)
    print(f"Original dataset size: {len(dataset):,}")
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        print(f"Training on limited dataset: {len(dataset):,} samples")
    else:
        print(f"Training on full dataset: {len(dataset):,} samples")

    val_size = max(1, int(0.01 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Training samples   : {len(train_ds):,}")
    print(f"Validation samples : {len(val_ds):,}")


    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = Transformer(
        embed_dim=args.embed_dim,
        src_vocab_size=tok_en.get_vocab_size(),
        target_vocab_size=tok_fr.get_vocab_size(),
        seq_length=args.seq_len,
        num_layers=args.num_layers,
        expansion_factor=args.expansion_factor,
        n_heads=args.n_heads,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    start_epoch = 0

    ckpt_path = f"{args.out_dir}/last.pt"
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        accumulation_steps = 2
        optimizer.zero_grad()
        for step, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:, :-1]
            trg_expected = trg[:, 1:]

            
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(src, trg_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), trg_expected.reshape(-1))

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch} "
                    f"step {step}/{len(train_loader)} "
                    f"loss {(loss.item() * accumulation_steps):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.7f}"
                )
        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                trg_input = trg[:, :-1]
                trg_expected = trg[:, 1:]
                logits = model(src, trg_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), trg_expected.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"== epoch {epoch} done in {time.time()-epoch_start:.1f}s "
              f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} ==")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{args.out_dir}/best_model.pt")
            print(f"New best model saved (val_loss={best_val_loss:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        }, ckpt_path)


    config = {
        "embed_dim": args.embed_dim, "seq_len": args.seq_len, "num_layers": args.num_layers,
        "expansion_factor": args.expansion_factor, "n_heads": args.n_heads,
        "src_vocab_size": tok_en.get_vocab_size(), "target_vocab_size": tok_fr.get_vocab_size(),
        "src_pad_idx": src_pad_idx, "trg_pad_idx": trg_pad_idx,
        "sos_idx": tok_fr.token_to_id("<sos>"), "eos_idx": tok_fr.token_to_id("<eos>"),
    }
    with open(f"{args.out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training complete. Artifacts in {args.out_dir}/: best_model.pt, last.pt, config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--tok_dir", default="tokenizers")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(args)