import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, x):
        return self.embed(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super().__init__()
        self.embed_dim = embed_model_dim
        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].detach()
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = embed_dim // n_heads

        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)

        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)

        if mask is not None:
            # mask: broadcastable bool tensor, True = keep, False = mask out
            product = product.masked_fill(mask == 0, torch.finfo(product.dtype).min)

        product = product / math.sqrt(self.single_head_dim)
        scores = F.softmax(product, dim=-1)
        scores = torch.matmul(scores, v)

        concat = scores.transpose(1, 2).contiguous().view(
            batch_size, seq_length_query, self.single_head_dim * self.n_heads
        )
        return self.out(concat)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value, mask=None, residual_base=None):
        # residual_base defaults to `value` (correct for self-attention, where
        # key=query=value). Cross-attention passes the decoder's own state as
        # residual_base explicitly, since there key != query != value.
        if residual_base is None:
            residual_base = value
        attention_out = self.attention(key, query, value, mask=mask)
        attention_residual_out = attention_out + residual_base
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))
        return norm2_out


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2,
                expansion_factor=4, n_heads=8, pad_idx=0):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, embed_dim, pad_idx=pad_idx)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask=None):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out, mask=src_mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads=n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, enc_out, x, trg_mask, cross_mask=None):
        # masked self-attention on target (causal + target-padding mask)
        attention = self.attention(x, x, x, mask=trg_mask)
        query = self.dropout(self.norm(attention + x))
        # cross-attention: Q from decoder, K and V both from encoder output.
        # residual_base=query keeps the residual stream on the decoder side,
        # not the encoder side (which would silently mix the two sequences).
        out = self.transformer_block(enc_out, query, enc_out, mask=cross_mask, residual_base=query)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2,
                expansion_factor=4, n_heads=8, pad_idx=0):
        super().__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, trg_mask, cross_mask=None):
        x = self.word_embedding(x)
        x = self.position_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(enc_out, x, trg_mask, cross_mask=cross_mask)
        # IMPORTANT: return raw logits. Apply softmax only at inference,
        # never before nn.CrossEntropyLoss (which does log-softmax internally).
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length,
                num_layers=2, expansion_factor=4, n_heads=8,
                src_pad_idx=0, trg_pad_idx=0):
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = TransformerEncoder(
            seq_length, src_vocab_size, embed_dim, num_layers, expansion_factor, n_heads, pad_idx=src_pad_idx
        )
        self.decoder = TransformerDecoder(
            target_vocab_size, embed_dim, seq_length, num_layers, expansion_factor, n_heads, pad_idx=trg_pad_idx
        )

    def make_src_mask(self, src):
        # (batch, 1, 1, src_len) -- True where token is real, False where padding
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # (b,1,1,trg_len)
        causal_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,trg_len,trg_len)
        trg_mask = pad_mask & causal_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src, src_mask=src_mask)
        outputs = self.decoder(trg, enc_out, trg_mask, cross_mask=src_mask)
        return outputs  # raw logits: (batch, trg_len, target_vocab_size)

    @torch.no_grad()
    def greedy_decode(self, src, sos_idx, eos_idx, max_len=100):
        self.eval()
        src_mask = self.make_src_mask(src)
        enc_out = self.encoder(src, src_mask=src_mask)
        batch_size = src.size(0)
        trg = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)

        for _ in range(max_len):
            trg_mask = self.make_trg_mask(trg)
            out = self.decoder(trg, enc_out, trg_mask, cross_mask=src_mask)
            next_token = out[:, -1, :].argmax(-1, keepdim=True)
            trg = torch.cat([trg, next_token], dim=1)
            if (next_token == eos_idx).all():
                break
        return trg