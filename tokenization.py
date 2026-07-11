"""
tokenization.py
-----------------
Two jobs, both streamed from disk in chunks so Data.csv (8GB) never sits
fully in RAM:

  1. train_tokenizers()  -> tokenizers/tokenizer_en.json, tokenizer_fr.json
  2. preprocess()         -> data/processed/src.npy, trg.npy
                             (memory-mapped, fixed-length, int32)

Run once before train.py:
    python tokenization.py --csv Data.csv --seq_len 128
"""
import argparse
import os

import numpy as np
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]



# Step 1: train BPE tokenizers by streaming the CSV in chunks

def line_iterator(csv_path, column, chunksize=50_000):
    for chunk in pd.read_csv(csv_path, usecols=[column], chunksize=chunksize, dtype=str):
        for val in chunk[column].fillna("").tolist():
            yield val






def train_one_tokenizer(csv_path, column, vocab_size, out_path):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(line_iterator(csv_path, column), trainer=trainer)
    tokenizer.save(out_path)
    print(f"Saved tokenizer for '{column}' -> {out_path} (vocab_size={tokenizer.get_vocab_size()})")


def train_tokenizers(csv_path, src_col, trg_col, vocab_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_one_tokenizer(csv_path, src_col, vocab_size, f"{out_dir}/tokenizer_en.json")
    train_one_tokenizer(csv_path, trg_col, vocab_size, f"{out_dir}/tokenizer_fr.json")






# Step 2: pre-tokenize the whole CSV into fixed-length memmapped arrays

def encode_line(tokenizer, text, seq_len, pad_id, sos_id, eos_id):
    ids = tokenizer.encode(text).ids
    ids = ids[: seq_len - 2]
    ids = [sos_id] + ids + [eos_id]
    if len(ids) < seq_len:
        ids = ids + [pad_id] * (seq_len - len(ids))
    return ids






def preprocess(csv_path, src_col, trg_col, tok_dir, seq_len, out_dir,
                chunksize=50_000, max_len_ratio=3.0):
    os.makedirs(out_dir, exist_ok=True)
    tok_en = Tokenizer.from_file(f"{tok_dir}/tokenizer_en.json")
    tok_fr = Tokenizer.from_file(f"{tok_dir}/tokenizer_fr.json")

    pad_en, sos_en, eos_en = tok_en.token_to_id("<pad>"), tok_en.token_to_id("<sos>"), tok_en.token_to_id("<eos>")
    pad_fr, sos_fr, eos_fr = tok_fr.token_to_id("<pad>"), tok_fr.token_to_id("<sos>"), tok_fr.token_to_id("<eos>")

    # First pass: count rows that survive dropna + a basic length-ratio filter,
    # so the memmap can be preallocated to the right size.
    n_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=[src_col, trg_col], chunksize=chunksize, dtype=str):
        chunk = chunk.dropna()
        lens_ok = (chunk[src_col].str.len() + 1) / (chunk[trg_col].str.len() + 1)
        chunk = chunk[(lens_ok > 1 / max_len_ratio) & (lens_ok < max_len_ratio)]
        n_rows += len(chunk)
    print(f"Total valid pairs after filtering: {n_rows}")

    src_mm = np.lib.format.open_memmap(f"{out_dir}/src.npy", mode="w+", dtype=np.int32, shape=(n_rows, seq_len))
    trg_mm = np.lib.format.open_memmap(f"{out_dir}/trg.npy", mode="w+", dtype=np.int32, shape=(n_rows, seq_len))

    idx = 0
    for chunk in pd.read_csv(csv_path, usecols=[src_col, trg_col], chunksize=chunksize, dtype=str):
        chunk = chunk.dropna()
        lens_ok = (chunk[src_col].str.len() + 1) / (chunk[trg_col].str.len() + 1)
        chunk = chunk[(lens_ok > 1 / max_len_ratio) & (lens_ok < max_len_ratio)]
        for en_text, fr_text in zip(chunk[src_col], chunk[trg_col]):
            src_mm[idx] = encode_line(tok_en, en_text, seq_len, pad_en, sos_en, eos_en)
            trg_mm[idx] = encode_line(tok_fr, fr_text, seq_len, pad_fr, sos_fr, eos_fr)
            idx += 1
        print(f"Processed {idx}/{n_rows}", end="\r")

    src_mm.flush()
    trg_mm.flush()
    print(f"\nSaved {out_dir}/src.npy and {out_dir}/trg.npy, shape=({n_rows},{seq_len})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="Data.csv")
    parser.add_argument("--src_col", default="en")
    parser.add_argument("--trg_col", default="fr")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--tok_dir", default="tokenizers")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    train_tokenizers(args.csv, args.src_col, args.trg_col, args.vocab_size, args.tok_dir)
    preprocess(args.csv, args.src_col, args.trg_col, args.tok_dir, args.seq_len, args.out_dir)