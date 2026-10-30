"""Hybrid BiGRU sentiment model (slide 9: architecture diagram).

Text branch:       Embedding -> BiGRU(128, all time steps) -> pooling -> Dropout -> Dense(64, ReLU)
Structured branch: one-hot categoricals + scaled numerics -> Dense(32, ReLU)   (optional)
Head:              Concatenate -> Dense(32, ReLU) -> Softmax(Negative / Neutral / Positive)
"""
import torch
import torch.nn as nn


class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, n_classes=3, embed_dim=128, hidden=128,
                 dropout=0.4, n_struct=0, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # GRU outputs for every step are mean- and max-pooled: 2 directions x 2 pools
        self.text_dense = nn.Sequential(nn.Linear(hidden * 4, 64), nn.ReLU())
        self.n_struct = n_struct
        if n_struct:
            self.struct_dense = nn.Sequential(nn.Linear(n_struct, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(64 + (32 if n_struct else 0), 32), nn.ReLU(),
            nn.Dropout(dropout / 2), nn.Linear(32, n_classes))

    use_packing = True  # set False on devices without packed-sequence support (DirectML)

    def _gru_unpacked(self, x, lengths):
        """The same bidirectional GRU (same weights, same result at every real position)
        written with plain matmul / sigmoid / tanh, for devices such as DirectML that lack
        PyTorch's fused GRU kernel and packed sequences. The backward direction reads each
        comment reversed (padding kept at the end), so both directions run forward in time
        together, then its outputs are flipped back."""
        B, T, _ = x.shape
        g = self.gru
        steps = torch.arange(T, device=x.device).unsqueeze(0)
        lens = lengths.unsqueeze(1)
        rev = torch.where(steps < lens, lens - 1 - steps, steps)

        def gather(t, idx):
            return t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.size(2)))

        inputs = torch.stack([x, gather(x, rev)])                              # (2, B, T, E)
        w_ih = torch.stack([g.weight_ih_l0, g.weight_ih_l0_reverse])            # (2, 3H, E)
        w_hh = torch.stack([g.weight_hh_l0, g.weight_hh_l0_reverse])            # (2, 3H, H)
        b_ih = torch.stack([g.bias_ih_l0, g.bias_ih_l0_reverse]).view(2, 1, 1, -1)
        b_hh = torch.stack([g.bias_hh_l0, g.bias_hh_l0_reverse]).unsqueeze(1)   # (2, 1, 3H)
        gi = torch.matmul(inputs, w_ih.transpose(1, 2).unsqueeze(1)) + b_ih    # (2, B, T, 3H)
        gi = gi.permute(2, 0, 1, 3).contiguous()                               # (T, 2, B, 3H)
        w_hh_t = w_hh.transpose(1, 2).contiguous()
        h = x.new_zeros(2, B, g.hidden_size)
        outs = []
        for t in range(T):
            gh = torch.baddbmm(b_hh, h, w_hh_t)
            i_r, i_z, i_n = gi[t].chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1 - z) * n + z * h
            outs.append(h)
        out = torch.stack(outs, dim=2)                                          # (2, B, T, H)
        return torch.cat([out[0], gather(out[1], rev)], dim=2)

    def forward(self, ids, lengths, struct=None):
        x = self.embedding(ids)
        if self.use_packing:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        else:
            out = self._gru_unpacked(x, lengths)
        # Float mask (1 = word, 0 = padding); DirectML cannot backpropagate through bool masks
        mask = (ids != 0).unsqueeze(-1).to(out.dtype)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = (out + (mask - 1) * 1e4).max(1).values
        h = self.text_dense(self.dropout(torch.cat([mean, mx], dim=1)))
        if self.n_struct:
            h = torch.cat([h, self.struct_dense(struct)], dim=1)
        return self.head(h)  # logits; softmax is applied at prediction time
