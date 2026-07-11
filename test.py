import json
import argparse

import torch
from tokenizers import Tokenizer

from model import Transformer


def load_model(checkpoint_dir, tok_dir, device):
    # Load configuration
    with open(f"{checkpoint_dir}/config.json", "r") as f:
        config = json.load(f)

    # Load tokenizers
    tok_en = Tokenizer.from_file(f"{tok_dir}/tokenizer_en.json")
    tok_fr = Tokenizer.from_file(f"{tok_dir}/tokenizer_fr.json")

    # Build model
    model = Transformer(
        embed_dim=config["embed_dim"],
        src_vocab_size=config["src_vocab_size"],
        target_vocab_size=config["target_vocab_size"],
        seq_length=config["seq_len"],
        num_layers=config["num_layers"],
        expansion_factor=config["expansion_factor"],
        n_heads=config["n_heads"],
        src_pad_idx=config["src_pad_idx"],
        trg_pad_idx=config["trg_pad_idx"],
    ).to(device)

    # Load trained weights
    model.load_state_dict(
        torch.load(
            f"{checkpoint_dir}/best_model.pt",
            map_location=device,
        )
    )

    model.eval()

    return model, tok_en, tok_fr, config


def encode_sentence(sentence, tokenizer, seq_len):
    pad = tokenizer.token_to_id("<pad>")
    sos = tokenizer.token_to_id("<sos>")
    eos = tokenizer.token_to_id("<eos>")

    ids = tokenizer.encode(sentence).ids
    ids = ids[:seq_len - 2]
    ids = [sos] + ids + [eos]

    if len(ids) < seq_len:
        ids += [pad] * (seq_len - len(ids))

    return torch.tensor(ids).unsqueeze(0)


def decode_tokens(tokens, tokenizer):
    pad = tokenizer.token_to_id("<pad>")
    sos = tokenizer.token_to_id("<sos>")
    eos = tokenizer.token_to_id("<eos>")

    ids = tokens.tolist()

    result = []

    for token in ids:

        if token == eos:
            break

        if token in (pad, sos):
            continue

        result.append(token)

    return tokenizer.decode(result)


def translate(sentence, model, tok_en, tok_fr, config, device):

    src = encode_sentence(
        sentence,
        tok_en,
        config["seq_len"],
    ).to(device)

    output = model.greedy_decode(
        src,
        sos_idx=config["sos_idx"],
        eos_idx=config["eos_idx"],
        max_len=config["seq_len"],
    )

    prediction = decode_tokens(output[0], tok_fr)

    return prediction


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints"
    )

    parser.add_argument(
        "--tok_dir",
        default="tokenizers"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, tok_en, tok_fr, config = load_model(
        args.checkpoint_dir,
        args.tok_dir,
        device,
    )

    print("=" * 60)
    print("English → French Translator")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:

        sentence = input("\nEnglish: ")

        if sentence.lower() == "quit":
            break

        prediction = translate(
            sentence,
            model,
            tok_en,
            tok_fr,
            config,
            device,
        )

        print("French :", prediction)


if __name__ == "__main__":
    main()