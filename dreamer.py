import os
import glob
import copy  # noqa: F401
import time
import queue
import threading
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, kl_divergence, OneHotCategoricalStraightThrough, OneHotCategorical
from torch.distributions.utils import probs_to_logits
from torch.nn.attention import sdpa_kernel, SDPBackend

from PokemonRedEnv import (LTM_REWARD_DIM,
                           GRID_DIM, GRID_CENTER_INDEX, MAP_NAMES)

IMAGE_SIZE = 64

torch.set_float32_matmul_precision('high')


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _mlp2(in_dim, out_dim, hidden=1024, act=nn.ELU, norm_out=True):
    """Two-layer MLP (one hidden layer). `hidden` controls the MLP width."""
    layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), act(), nn.Linear(hidden, out_dim)]
    if norm_out:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def _mlp3(in_dim, out_dim, hidden=1024, act=nn.SiLU):
    """Three-hidden-layer MLP (four Linear layers). `hidden` controls the MLP width."""
    d = hidden
    return nn.Sequential(
        nn.Linear(in_dim, d), nn.LayerNorm(d), act(),
        nn.Linear(d, d), nn.LayerNorm(d), act(),
        nn.Linear(d, d), nn.LayerNorm(d), act(),
        nn.Linear(d, out_dim),
    )


# ==========================================================
# SCALAR ENCODINGS
# ==========================================================
class TwoHotEncoding:
    """Symlog two-hot encoding (DreamerV3) for value/reward estimation."""
    def __init__(self, min=-10, max=10, num_bins=255, device="cuda"):
        self.device = device
        self.num_bins = num_bins
        self.bins = torch.linspace(min, max, num_bins, device=device)
        self.bin_values = symexp(self.bins)

    def encode(self, x):
        x = symlog(x)
        x = torch.clamp(x, self.bins[0], self.bins[-1])
        pos = (x - self.bins[0]) / (self.bins[1] - self.bins[0])
        low = torch.clamp(torch.floor(pos).long(), 0, self.num_bins - 2)
        high = low + 1
        weight_high = pos - low
        weight_low = 1.0 - weight_high
        two_hot = torch.zeros(*x.shape, self.num_bins, device=self.device)
        two_hot.scatter_(-1, low.unsqueeze(-1), weight_low.unsqueeze(-1))
        two_hot.scatter_(-1, high.unsqueeze(-1), weight_high.unsqueeze(-1))
        return two_hot

    def decode(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * self.bin_values, dim=-1, keepdim=True)


# ==========================================================
# IMAGE ENCODER / DECODER
# ==========================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),  # LayerNorm equivalent for images
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderImage(nn.Module):
    def __init__(self, output_size=1024, depth=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, depth, kernel_size=4, stride=2, padding=1), nn.ELU(), ResBlock(depth),                 # 64 -> 32
            nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=1), nn.ELU(), ResBlock(depth * 2),     # 32 -> 16
            nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1), nn.ELU(), ResBlock(depth * 4), # 16 -> 8
            nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1), nn.ELU(),                      # 8 -> 4
            nn.Flatten(),
            nn.Linear(depth * 8 * 4 * 4, output_size),
            nn.LayerNorm(output_size),
        )
        # Lay the conv stack out for channels_last so cuDNN picks its fast kernels.
        self.layers = self.layers.to(memory_format=torch.channels_last)

    def forward(self, x):
        # Match input layout to the weights (no-op on CPU, faster convs on CUDA).
        x = x.contiguous(memory_format=torch.channels_last)
        return self.layers(x)


class Decoder(nn.Module):
    """Image reconstruction decoder: full state -> observation. Trained jointly with
    the world model; its reconstruction loss is the main representation signal."""
    def __init__(self, input_size, depth=32):
        super().__init__()
        self.depth = depth
        self.linear = nn.Linear(input_size, depth * 8 * 4 * 4)
        self.net = nn.Sequential(
            ResBlock(depth * 8),
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, stride=2, padding=1), nn.ELU(),  # 4 -> 8
            ResBlock(depth * 4),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, stride=2, padding=1), nn.ELU(),  # 8 -> 16
            ResBlock(depth * 2),
            nn.ConvTranspose2d(depth * 2, depth, 4, stride=2, padding=1), nn.ELU(),      # 16 -> 32
            nn.ConvTranspose2d(depth, 3, 4, stride=2, padding=1),                        # 32 -> 64
        )
        self.net = self.net.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.depth * 8, 4, 4)
        x = x.contiguous(memory_format=torch.channels_last)
        return self.net(x)


# ==========================================================
# AUXILIARY ENCODERS (state -> feature) AND PREDICTORS (latent -> state)
# ==========================================================
class TeamItemEncoder(nn.Module):
    """Combined party-levels + item-counts encoder (single encoder, feature #3)."""
    def __init__(self, in_dim=8, out_dim=256, hidden=1024):
        super().__init__()
        self.net = _mlp2(in_dim, out_dim, hidden=hidden)

    def forward(self, x):
        return self.net(x)


class TeamItemPredictor(nn.Module):
    """Combined party-levels + item-counts predictor (single decoder, feature #3)."""
    def __init__(self, input_size, out_dim=8, hidden=1024):
        super().__init__()
        self.net = _mlp2(input_size, out_dim, hidden=hidden, act=nn.SiLU, norm_out=False)

    def forward(self, x):
        return self.net(x)


class LongTermMemoryEncoder(nn.Module):
    """Whole-game long-term reward-memory (event flags) encoder (feature #2)."""
    def __init__(self, in_dim=LTM_REWARD_DIM, out_dim=512, hidden=1024):
        super().__init__()
        self.net = _mlp2(in_dim, out_dim, hidden=hidden)

    def forward(self, x):
        return self.net(x)


class LongTermMemoryPredictor(nn.Module):
    """Whole-game long-term reward-memory predictor (multi-label logits)."""
    def __init__(self, input_size, out_dim=LTM_REWARD_DIM, hidden=1024):
        super().__init__()
        self.net = _mlp2(input_size, out_dim, hidden=hidden, act=nn.SiLU, norm_out=False)
        # Init flags off so nothing is hallucinated initially.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, x):
        return self.net(x)


class GridEncoder(nn.Module):
    """Local 5x5 explored-grid encoder (agent-centered exploration map, feature #grid)."""
    def __init__(self, in_dim=GRID_DIM, out_dim=128, hidden=1024):
        super().__init__()
        self.net = _mlp2(in_dim, out_dim, hidden=hidden)

    def forward(self, x):
        return self.net(x)


class GridPredictor(nn.Module):
    """Local 5x5 explored-grid predictor; the dream gates tile curiosity on its
    CENTER cell (last layer zero-init so every cell starts at p=0.5)."""
    def __init__(self, input_size, out_dim=GRID_DIM, hidden=1024):
        super().__init__()
        self.net = _mlp2(input_size, out_dim, hidden=hidden, act=nn.SiLU, norm_out=False)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ==========================================================
# REWARD / CURIOSITY HEADS
# ==========================================================
class RewardPredictor(nn.Module):
    def __init__(self, inputsize, num_bins=255, mlp_dim=1024):
        super().__init__()
        self.transform = _mlp3(inputsize, num_bins, hidden=mlp_dim, act=nn.SiLU)
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)

    def forward(self, x):
        return self.transform(x)


class CuriosityPredictor(nn.Module):
    """Learned curiosity head (two hidden layers); biased toward 0 at init."""
    def __init__(self, input_size, num_bins=255, mlp_dim=1024):
        super().__init__()
        d = mlp_dim
        self.net = nn.Sequential(
            nn.Linear(input_size, d), nn.LayerNorm(d), nn.SiLU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.SiLU(),
            nn.Linear(d, num_bins),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            # Symlog two-hot: value 0 sits at the center bin, so bias toward it.
            self.net[-1].bias[num_bins // 2] = 10.0

    def forward(self, x):
        return self.net(x)


# ==========================================================
# TSSM CORE (Transformer State-Space Model)
# ==========================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rope_cos_sin(pos, half_dim, device, base=10000.0):
    """cos/sin tables for rotary embeddings at (possibly per-row) positions `pos`."""
    inv_freq = base ** (-torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim)
    angles = pos.to(torch.float32).unsqueeze(-1) * inv_freq
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x, cos, sin):
    """Rotate the head dimension of x [..., Dh]; cos/sin broadcast to x[..., :Dh/2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class TSSMCache:

    def __init__(self, n_layers, batch, max_len, n_kv_heads, head_dim, device, dtype=torch.float32):
        self.k = torch.zeros(n_layers, batch, max_len, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.pos = torch.zeros(batch, dtype=torch.long, device=device)
        self.key_pos = torch.full((batch, max_len), -1, dtype=torch.long, device=device)
        self.max_len = max_len

    def reset_rows(self, rows):
        """Clear the context of the given row(s) (e.g. on environment reset)."""
        self.pos[rows] = 0
        self.key_pos[rows] = -1

    def prefill(self, kv, row_idx, lengths):
        """Load real-history context: for each cache row i, the first `lengths[i]`
        tokens of source sequence `row_idx[i]` become its attention context.
        `kv` is a per-layer list of (k, v), each [B, T, H_kv, Dh] (RoPE already applied)."""
        T = kv[0][0].shape[1]
        assert T <= self.max_len, "prefill longer than cache"
        for li, (k, v) in enumerate(kv):
            self.k[li, :, :T] = k[row_idx]
            self.v[li, :, :T] = v[row_idx]
        lengths = lengths.to(self.pos.device, dtype=torch.long)
        steps = torch.arange(T, device=self.pos.device)
        kp = steps[None, :].expand(lengths.shape[0], T).clone()
        kp[steps[None, :] >= lengths[:, None]] = -1
        self.key_pos[:, :T] = kp
        self.key_pos[:, T:] = -1
        self.pos = lengths.clone()


class TSSMBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> causal attention (RoPE, QKNorm, GQA)
    -> residual -> RMSNorm -> SwiGLU -> residual."""
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.groups = n_heads // n_kv_heads
        hd = self.head_dim
        self.norm_attn = RMSNorm(d_model)
        self.norm_mlp = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(n_heads * hd, d_model, bias=False)
        self.q_norm = RMSNorm(hd)   # QKNorm for attention stability (Dreamer 4)
        self.k_norm = RMSNorm(hd)
        self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)

    def _mlp(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward_train(self, x, cos, sin, attn_mask):
        """Parallel forward over a full sequence with a windowed-causal mask.
        x: [B, T, d]; attn_mask: [T, T] bool (True = may attend).
        Returns the block output and the (post-RoPE) K/V for KV-cache reuse."""
        B, T, _ = x.shape
        h = self.norm_attn(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(self.q_norm(q), cos, sin)
        k = _apply_rope(self.k_norm(k), cos, sin)
        # GQA: expand KV heads to match query heads for the fused kernel.
        kr = k.repeat_interleave(self.groups, dim=1)
        vr = v.repeat_interleave(self.groups, dim=1)
        attn = F.scaled_dot_product_attention(q, kr, vr, attn_mask=attn_mask)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(B, T, -1))
        x = x + self._mlp(self.norm_mlp(x))
        return x, (k.transpose(1, 2), v.transpose(1, 2))  # cache layout [B, T, H_kv, Dh]

    def forward_step(self, x, k_cache, v_cache, slot, key_mask, cos, sin):
        """Single-token decode through the KV cache (imagination / acting).
        x: [N, d]; k_cache/v_cache: [N, L, H_kv, Dh]; slot: [N]; key_mask: [N, L]."""
        N = x.shape[0]
        h = self.norm_attn(x)
        q = self.q_proj(h).view(N, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(N, self.n_kv_heads, self.head_dim)
        v = self.v_proj(h).view(N, self.n_kv_heads, self.head_dim)
        q = _apply_rope(self.q_norm(q), cos, sin)
        k = _apply_rope(self.k_norm(k), cos, sin)
        rows = torch.arange(N, device=x.device)
        k_cache[rows, slot] = k
        v_cache[rows, slot] = v
        qg = q.view(N, self.n_kv_heads, self.groups, self.head_dim)
        scores = torch.einsum('nkgd,nlkd->nkgl', qg, k_cache) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~key_mask[:, None, None, :], float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum('nkgl,nlkd->nkgd', probs, v_cache).reshape(N, -1)
        x = x + self.o_proj(out)
        x = x + self._mlp(self.norm_mlp(x))
        return x


class TSSM(nn.Module):
    def __init__(self, d_model=1024, latentSize=1600, actionSize=6,
                 n_layers=4, n_heads=8, n_kv_heads=2, ffn_hidden=2816, window=192,
                 z_rows=40, z_cols=40, embed_hidden=None):
        super().__init__()
        assert z_rows * z_cols == latentSize, "z_rows * z_cols must equal latentSize"
        self.d_model = d_model
        self.recurrentSize = d_model  # output size (kept name-compatible with the GRU model)
        self.latentSize = latentSize
        self.actionSize = actionSize
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.window = window
        # --- Simple MLP tokenizer for u_t = (z_t, a_t) ---
        # Concatenate the flattened one-hot latent with the action, then project
        # to d_model through a 2-layer MLP.
        embed_hidden = embed_hidden if embed_hidden is not None else ffn_hidden
        self.token_mlp = nn.Sequential(
            nn.Linear(latentSize + actionSize, embed_hidden),
            nn.SiLU(),
            nn.Linear(embed_hidden, d_model),
        )
        self.embed_norm = RMSNorm(d_model)

        self.blocks = nn.ModuleList(
            [TSSMBlock(d_model, n_heads, n_kv_heads, ffn_hidden) for _ in range(n_layers)])
        self.norm_out = RMSNorm(d_model)

    def embed(self, latent, action):
        """Token embedding for u_t = (z_t, a_t): concatenate the flattened one-hot
        latent [..., Z] with the action [..., A] and project to d_model with a
        simple MLP. Returns [..., d_model]."""
        x = self.token_mlp(torch.cat((latent, action), -1))
        return self.embed_norm(x)

    def make_cache(self, batch, max_len, device, dtype=torch.float32):
        return TSSMCache(len(self.blocks), batch, max_len, self.n_kv_heads,
                         self.head_dim, device, dtype)

    def forward_sequence(self, latents, actions, return_kv=False):
        """Parallel training pass. latents [B, T, Z], actions [B, T, A] are the
        tokens u_t = (z_t, a_t); output [:, t] is h_{t+1}. Optionally returns the
        detached per-layer K/V so imagination can reuse them as prefix context."""
        x = self.embed(latents, actions)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        cos, sin = _rope_cos_sin(pos, self.head_dim // 2, x.device)
        cos, sin = cos[None, None], sin[None, None]
        # Banded causal mask: attend to self and up to `window - 1` steps back.
        rel = pos[:, None] - pos[None, :]
        attn_mask = (rel >= 0) & (rel < self.window)
        kv = [] if return_kv else None
        for blk in self.blocks:
            x, layer_kv = blk.forward_train(x, cos, sin, attn_mask)
            if return_kv:
                kv.append((layer_kv[0].detach(), layer_kv[1].detach()))
        return self.norm_out(x), kv

    def forward_step(self, latent, action, cache):
        """Append one (z, a) token per row and return the next deterministic state
        h. latent [N, Z], action [N, A]; the KV cache is updated in place."""
        x = self.embed(latent, action)
        pos = cache.pos
        slot = pos % cache.max_len
        rows = torch.arange(x.shape[0], device=x.device)
        cache.key_pos[rows, slot] = pos  # current token becomes an attendable key
        cos, sin = _rope_cos_sin(pos, self.head_dim // 2, x.device)
        cos, sin = cos[:, None], sin[:, None]
        # Same sliding window as training, over absolute key positions.
        key_mask = (cache.key_pos >= 0) & (cache.key_pos > (pos - self.window)[:, None])
        for li, blk in enumerate(self.blocks):
            x = blk.forward_step(x, cache.k[li], cache.v[li], slot, key_mask, cos, sin)
        cache.pos = pos + 1
        return self.norm_out(x)


class PriorNet(nn.Module):
    def __init__(self, recurrentSize=512, rows=16, cols=16, mlp_dim=1024):
        super().__init__()
        self.recurrentSize = recurrentSize
        self.latentSize = rows * cols
        self.rows = rows
        self.cols = cols
        self.trasform = _mlp3(recurrentSize, rows * cols, hidden=mlp_dim, act=nn.SiLU)

    def forward(self, RecurrentState, unimix=True):
        """unimix=True mixes 1% uniform into the categorical (training: keeps KL
        gradients alive). unimix=False samples the raw distribution (imagination:
        avoids injecting random latent flips that compound over the rollout)."""
        rawLogits = self.trasform(RecurrentState)
        rawProbabilities = rawLogits.view(-1, self.rows, self.cols).softmax(-1)
        if unimix:
            confusion = torch.ones_like(rawProbabilities) / self.cols
            probabilities = 0.99 * rawProbabilities + 0.01 * confusion
        else:
            probabilities = rawProbabilities
        logits = probs_to_logits(probabilities)
        sample = Independent(OneHotCategoricalStraightThrough(probs=probabilities), 1).rsample().view(-1, self.latentSize)
        return sample, logits


class PosteriorNet(nn.Module):
    def __init__(self, inputSize=1536, rows=16, cols=16, mlp_dim=1024):
        super().__init__()
        self.inputSize = inputSize
        self.rows = rows
        self.cols = cols
        self.latentSize = rows * cols
        self.trasform = _mlp3(inputSize, self.latentSize, hidden=mlp_dim, act=nn.SiLU)

    def forward(self, InputState):
        rawLogits = self.trasform(InputState)
        rawProbabilities = rawLogits.view(-1, self.rows, self.cols).softmax(-1)
        confusion = torch.ones_like(rawProbabilities) / self.cols
        probabilities = 0.99 * rawProbabilities + 0.01 * confusion
        logits = probs_to_logits(probabilities)
        sample = Independent(OneHotCategoricalStraightThrough(probs=probabilities), 1).rsample().view(-1, self.rows * self.cols)
        return sample, logits


# ==========================================================
# ACTOR / CRITIC
# ==========================================================
class Actor(nn.Module):
    def __init__(self, action_dim, device, concatenated_dim=768, mlp_dim=1024):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.concatenated_dim = concatenated_dim
        self.net = _mlp3(concatenated_dim, action_dim, hidden=mlp_dim, act=nn.SiLU)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        raw_logits = self.net(x)
        probs = torch.softmax(raw_logits, dim=-1)
        uniform = torch.ones_like(probs) / self.action_dim
        mixed_probs = 0.99 * probs + 0.01 * uniform
        dist = OneHotCategorical(probs=mixed_probs)
        action_onehot = dist.sample()
        return action_onehot, dist.log_prob(action_onehot), dist.entropy()


class Critic(nn.Module):
    def __init__(self, inputSize, bins=255, mlp_dim=1024):
        super().__init__()
        self.transform = _mlp3(inputSize, bins, hidden=mlp_dim, act=nn.SiLU)
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)

    def forward(self, x):
        return self.transform(x)


# ==========================================================
# REPLAY BUFFER
# ==========================================================
class Buffer(object):
    def __init__(self, device, capacity=800000, actionSize=6,
                 ltm_reward_dim=LTM_REWARD_DIM,
                 item_dim=2, team_level_dim=6, num_envs=4, grid_dim=GRID_DIM,
                 reward_sample_prob=0.0, reward_threshold=50.0, recent_sample_prob=0.0):
        self.device = device
        self.capacity = capacity
        self.num_envs = num_envs
        # Reward-guaranteed sampling: per-sequence chance that the drawn window is
        # forced to contain a transition with sparse reward >= reward_threshold.
        self.reward_sample_prob = reward_sample_prob
        self.reward_threshold = reward_threshold
        # Recency sampling: per-sequence chance to draw from the freshest data
        # (the last `num_envs` completed episodes, i.e. one collection round).
        self.recent_sample_prob = recent_sample_prob
        self._recent_lengths = deque(maxlen=num_envs)  # lengths of the last num_envs episodes
        self._since_episode_start = 0                  # transitions added since last end_episode()
        self._pin = torch.cuda.is_available()
        self.observations = torch.empty((capacity, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8, device='cpu')
        self.ltm_rewards = torch.empty((capacity, ltm_reward_dim), dtype=torch.uint8, device='cpu')
        self.grids = torch.empty((capacity, grid_dim), dtype=torch.uint8, device='cpu')
        self.item_counts = torch.empty((capacity, item_dim), dtype=torch.float32, device='cpu')
        self.team_levels = torch.empty((capacity, team_level_dim), dtype=torch.float32, device='cpu')
        self.actions = torch.empty((capacity, actionSize), dtype=torch.float32, device='cpu')
        self.sparse_rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.standard_rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.curiosities = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.tier_events = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.episode_ids = torch.zeros((capacity,), dtype=torch.long, device='cpu')
        self._episode_counter = 0

        self.index = 0
        self.full = False

    def add(self, observation, ltm_reward, grid, item_count, team_level, action,
            sparse_reward, standard_reward, curiosity, tier_event):
        self.observations[self.index] = torch.as_tensor(observation, dtype=torch.uint8)
        self.ltm_rewards[self.index] = torch.as_tensor(ltm_reward, dtype=torch.uint8)
        self.grids[self.index] = torch.as_tensor(grid, dtype=torch.uint8)
        self.item_counts[self.index] = torch.as_tensor(item_count, dtype=torch.float32)
        self.team_levels[self.index] = torch.as_tensor(team_level, dtype=torch.float32)
        self.actions[self.index] = torch.as_tensor(action, dtype=torch.float32)
        self.sparse_rewards[self.index] = torch.as_tensor(sparse_reward, dtype=torch.float32)
        self.standard_rewards[self.index] = torch.as_tensor(standard_reward, dtype=torch.float32)
        self.curiosities[self.index] = torch.as_tensor(curiosity, dtype=torch.float32)
        self.tier_events[self.index] = torch.as_tensor(tier_event, dtype=torch.float32)
        self.episode_ids[self.index] = self._episode_counter

        self.index = (self.index + 1) % self.capacity
        self.full = self.full or (self.index == 0)
        self._since_episode_start += 1

    def end_episode(self):
        """Mark an episode boundary; call after flushing one episode's transitions."""
        self._episode_counter += 1
        if self._since_episode_start > 0:
            self._recent_lengths.append(self._since_episode_start)
        self._since_episode_start = 0

    def _recent_span(self):
        """Transitions covered by the last `num_envs` completed episodes (they sit
        contiguously behind the write head, since episodes are flushed whole)."""
        return int(sum(self._recent_lengths))

    def _same_episode(self, rows):
        """rows: [num, T] index matrix -> bool [num], True if the whole window lies
        within one episode. Episode ids are monotone in write order, so equal ids at
        the two endpoints imply the interior is uniform too (T << capacity). This
        also catches windows that wrap across the write head into stale data."""
        return self.episode_ids[rows[:, 0]] == self.episode_ids[rows[:, -1]]

    def sample(self, batchSize, sequenceSize):
        N = self.capacity if self.full else self.index
        if N < sequenceSize:
            return None

        limit = self.capacity if self.full else (self.index - sequenceSize + 1)
        if limit <= 0:
            return None
        seq_offsets = torch.arange(sequenceSize).reshape(1, -1)

        def draw_uniform(n):
            return (torch.randint(0, limit, (n, 1)) + seq_offsets) % self.capacity

        # Recency sampling: the last `num_envs` episodes occupy the `span`
        # transitions right behind the write head; draw window starts inside it.
        span = min(self._recent_span(), N)
        use_recent = self.recent_sample_prob > 0.0 and span >= sequenceSize

        def draw_recent(n):
            starts = torch.randint(0, span - sequenceSize + 1, (n, 1))
            return (self.index - span + starts + seq_offsets) % self.capacity

        recent_mask = (torch.rand(batchSize) < self.recent_sample_prob) if use_recent \
            else torch.zeros(batchSize, dtype=torch.bool)

        # Reward-guaranteed sampling: force some windows to contain a transition with
        # sparse reward >= reward_threshold. The reward step is picked uniformly among
        # all such transitions, and its position inside the window is uniform too.
        reward_positions = None
        if self.reward_sample_prob > 0.0:
            reward_positions = (self.sparse_rewards[:N, 0] >= self.reward_threshold).nonzero(as_tuple=True)[0]
            if reward_positions.numel() == 0:
                reward_positions = None
        reward_mask = (torch.rand(batchSize) < self.reward_sample_prob) if reward_positions is not None \
            else torch.zeros(batchSize, dtype=torch.bool)
        recent_mask &= ~reward_mask  # reward guarantee takes precedence over recency

        def draw_reward(n):
            r = reward_positions[torch.randint(0, reward_positions.numel(), (n,))]
            offset = torch.randint(0, sequenceSize, (n,))  # reward lands at this slot in the window
            if self.full:
                starts = (r - offset) % self.capacity
            else:
                starts = (r - offset).clamp(min=0, max=self.index - sequenceSize)
            return (starts.reshape(-1, 1) + seq_offsets) % self.capacity

        rows = draw_uniform(batchSize)
        if use_recent and recent_mask.any():
            rows[recent_mask] = draw_recent(int(recent_mask.sum()))
        if reward_mask.any():
            rows[reward_mask] = draw_reward(int(reward_mask.sum()))
        for _ in range(8):  # redraw windows that cross an episode boundary
            bad = ~self._same_episode(rows)
            if not bad.any():
                break
            bad_reward = bad & reward_mask
            bad_recent = bad & recent_mask
            bad_uniform = bad & ~recent_mask & ~reward_mask
            if bad_reward.any():
                rows[bad_reward] = draw_reward(int(bad_reward.sum()))
            if bad_recent.any():
                rows[bad_recent] = draw_recent(int(bad_recent.sum()))
            if bad_uniform.any():
                rows[bad_uniform] = draw_uniform(int(bad_uniform.sum()))
        # Stragglers that still cross a boundary (e.g. reward too close to an episode
        # edge): fall back to plain uniform windows rather than train across a reset.
        bad = ~self._same_episode(rows)
        if bad.any():
            rows[bad] = draw_uniform(int(bad.sum()))

        sampleIndex = rows.long()
        batch = {
            "observations":       self.observations[sampleIndex],   # uint8
            "ltm_rewards":        self.ltm_rewards[sampleIndex],     # uint8
            "grids":              self.grids[sampleIndex],           # uint8
            "item_counts":        self.item_counts[sampleIndex],
            "team_levels":        self.team_levels[sampleIndex],
            "actions":            self.actions[sampleIndex],
            "sparse_rewards":     self.sparse_rewards[sampleIndex],
            "standard_rewards":   self.standard_rewards[sampleIndex],
            "curiosities":        self.curiosities[sampleIndex],
            "tier_events":        self.tier_events[sampleIndex],
            "index":              sampleIndex,
        }

        if self._pin:
            batch = {k: v.pin_memory() for k, v in batch.items()}
        return batch

    def save(self, path):
        limit = self.capacity if self.full else self.index
        torch.save({
            'observations': self.observations[:limit],
            'ltm_rewards': self.ltm_rewards[:limit],
            'grids': self.grids[:limit],
            'item_counts': self.item_counts[:limit],
            'team_levels': self.team_levels[:limit],
            'actions': self.actions[:limit],
            'sparse_rewards': self.sparse_rewards[:limit],
            'standard_rewards': self.standard_rewards[:limit],
            'curiosities': self.curiosities[:limit],
            'tier_events': self.tier_events[:limit],
            'episode_ids': self.episode_ids[:limit],
            'index': self.index,
            'full': self.full,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        n = min(ckpt['observations'].shape[0], self.capacity)
        for name in ('observations', 'ltm_rewards', 'grids', 'item_counts',
                     'team_levels', 'actions', 'sparse_rewards', 'standard_rewards',
                     'curiosities', 'tier_events', 'episode_ids'):
            getattr(self, name)[:n] = ckpt[name][:n]  # KeyError = incompatible old buffer
        self.index = n % self.capacity
        self.full = (n == self.capacity)
        # Resume episode numbering above anything restored.
        self._episode_counter = int(self.episode_ids[:n].max().item()) + 1 if n > 0 else 0
        # Rebuild the recency window from the last num_envs episode ids on disk.
        self._recent_lengths.clear()
        self._since_episode_start = 0
        if n > 0:
            ids = self.episode_ids[:n]
            for eid in torch.unique(ids)[-self.num_envs:].tolist():
                self._recent_lengths.append(int((ids == eid).sum().item()))
        print(f"[*] Loaded buffer with {n} transitions (full={self.full}, write_head={self.index}, "
              f"recent_span={self._recent_span()})")

    def print_diagnostics(self):
        valid = self.capacity if self.full else self.index
        print("\n" + "=" * 50)
        print("          REPLAY BUFFER DIAGNOSTICS")
        print("=" * 50)
        print(f"  Buffer Fill : {valid:,} / {self.capacity:,} ({100.0 * valid / self.capacity:.2f}%)")
        print(f"  Active Environments : {self.num_envs}")
        print(f"  Recent Sampling : p={self.recent_sample_prob} | window {self._recent_span():,} steps "
              f"({len(self._recent_lengths)} episodes)")
        if self.reward_sample_prob > 0.0:
            n_rewards = int((self.sparse_rewards[:valid, 0] >= self.reward_threshold).sum().item())
            print(f"  Reward Windows : p={self.reward_sample_prob} | "
                  f"{n_rewards} transitions with sparse reward >= {self.reward_threshold}")
        print("=" * 50 + "\n")


# ==========================================================
# BACKGROUND BATCH PREFETCHER
# ==========================================================
class BatchPrefetcher:
    """Samples pinned CPU batches on a background thread so the gather overlaps GPU
    training. Runs only during the training phase; episode collection happens
    sequentially afterward, so no buffer writes occur while it samples."""

    def __init__(self, buffer, batch_size, sequence_size, depth=3):
        self.buffer = buffer
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self._q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            batch = self.buffer.sample(self.batch_size, self.sequence_size)
            if batch is None:
                time.sleep(0.01)
                continue
            # Block until there's room, but stay responsive to close().
            while not self._stop.is_set():
                try:
                    self._q.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def get(self):
        return self._q.get()

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=1.0)


# ==========================================================
# DREAMER
# ==========================================================
class Dreamer:
    def __init__(self, device, envs,
                 action_dim=6, recurrent_dim=1024, rows=40, cols=40,
                 tssm_layers=4, tssm_heads=8, tssm_kv_heads=2, tssm_ffn=2816,
                 context_length=192,
                 number_of_sequences=32, steps_per_sequence=256, dreams_per_sequence=1,
                 buffer_size=1500000,
                 team_dim=6, item_dim=2, curiosity_scale=0.25, mlp_dim=1024,
                 entropy_scale=0.0015,
                 teamitem_out=128, ltm_reward_out=512, grid_out=128,
                 dream_lead_steps=10, ltm_gate_threshold=0.4, grid_gate_threshold=0.5,
                 grid_zero_weight=5.0,
                 var_beta_dyn=0.5, var_beta_reg=0.1, var_warmup_steps=20000,
                 loss_norm_decay=0.99, reward_loss_weight=1.0,
                 continue_discount=0.998, pmpo_alpha=0.5,
                 reward_sample_prob=0.0, reward_threshold=50.0,
                 recent_sample_prob=0.0):

        # --- Dimensions / config ---
        self.device = device
        self.action_dim = action_dim
        self.recurrent_dim = recurrent_dim
        self.mlp_dim = mlp_dim
        self.rows = rows
        self.cols = cols
        self.latent_dim = rows * cols
        self.concatenated_dim = recurrent_dim + self.latent_dim
        self.total_num_episodes = 0
        self.total_num_steps = 0
        self.total_num_updates = 0

        # Auxiliary feature sizes feeding the posterior.
        self.image_out = 1024
        self.teamitem_out = teamitem_out
        self.ltm_reward_out = ltm_reward_out
        self.grid_out = grid_out
        self.team_dim = team_dim
        self.item_dim = item_dim
        self.enconder_output_size = (self.image_out + self.teamitem_out
                                     + self.ltm_reward_out
                                     + self.grid_out)

        self.entropy_scale = entropy_scale
        self.number_of_sequences = number_of_sequences
        self.steps_per_sequence = steps_per_sequence
        self.dreams_per_sequence = dreams_per_sequence  # dream starts sampled per replay sequence
        self.buffer_capacity = buffer_size
        self.curiosity_scale = curiosity_scale
        self.envs = envs

        # PMPO
        self.pmpo_alpha = pmpo_alpha

        # Whole-game LTM loss shaping (sparse multi-label targets).
        self.ltm_reward_pos_weight = torch.tensor(200.0, device=self.device)
        self.ltm_sparsity_weight = 0.5

        # Curiosity targets are sparse (mostly zero): up-weight samples with real
        # (non-zero) curiosity so the head is punished more for missing them.
        self.curiosity_pos_weight = 5.0

        self.dream_lead_steps = dream_lead_steps

        # Dream sparse-reward curiosity gating
        self.ltm_gate_threshold = ltm_gate_threshold
        self.grid_gate_threshold = grid_gate_threshold
        self.grid_zero_weight = grid_zero_weight

        # --- Encodings ---
        self.two_hot = TwoHotEncoding(device=self.device)

        # --- TSSM (transformer state-space model) ---
        self.context_length = context_length
        self.recurrentModel = TSSM(d_model=self.recurrent_dim, latentSize=self.latent_dim,actionSize=self.action_dim, n_layers=tssm_layers,n_heads=tssm_heads, n_kv_heads=tssm_kv_heads,ffn_hidden=tssm_ffn, window=context_length,z_rows=self.rows, z_cols=self.cols).to(self.device)
        self.posteriorNet = PosteriorNet(self.enconder_output_size, self.rows, self.cols, mlp_dim=self.mlp_dim).to(self.device)
        self.priorNet = PriorNet(self.recurrent_dim, self.rows, self.cols, mlp_dim=self.mlp_dim).to(self.device)

        # --- Encoders ---
        self.image_encoder = EncoderImage(output_size=self.image_out).to(self.device)
        self.teamitem_encoder = TeamItemEncoder(self.team_dim + self.item_dim, self.teamitem_out, hidden=self.mlp_dim).to(self.device)
        self.ltm_reward_encoder = LongTermMemoryEncoder(LTM_REWARD_DIM, self.ltm_reward_out, hidden=self.mlp_dim).to(self.device)
        self.grid_encoder = GridEncoder(GRID_DIM, self.grid_out, hidden=self.mlp_dim).to(self.device)

        # --- Predictors (latent -> observation components) ---
        self.sparseRewardPredictor = RewardPredictor(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.standardRewardPredictor = RewardPredictor(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.curiosityPredictor = CuriosityPredictor(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.teamitemPredictor = TeamItemPredictor(self.concatenated_dim, self.team_dim + self.item_dim, hidden=self.mlp_dim).to(self.device)
        self.ltm_reward_predictor = LongTermMemoryPredictor(self.concatenated_dim, LTM_REWARD_DIM, hidden=self.mlp_dim).to(self.device)
        self.grid_predictor = GridPredictor(self.concatenated_dim, GRID_DIM, hidden=self.mlp_dim).to(self.device)

        # --- Image reconstruction decoder (part of the world-model objective) ---
        self.decoder = Decoder(input_size=self.concatenated_dim).to(self.device)

        # --- Actor / Critics ---
        self.actor = Actor(self.action_dim, self.device, self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.critic = Critic(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.curiosity_critic = Critic(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)


        # Latent-space value-alignment regularization (Var) config.
        self.var_beta_dyn = var_beta_dyn
        self.var_beta_reg = var_beta_reg
        self.var_warmup_steps = var_warmup_steps  # gated on total_num_updates (persisted in checkpoints)
        self.reward_loss_weight = reward_loss_weight
        self.continue_discount = continue_discount

        # --- Buffer ---
        self.buffer = Buffer(
            device=self.device, capacity=self.buffer_capacity, actionSize=self.action_dim,
            ltm_reward_dim=LTM_REWARD_DIM,
            item_dim=self.item_dim, team_level_dim=self.team_dim,
            num_envs=len(envs), grid_dim=GRID_DIM,
            reward_sample_prob=reward_sample_prob, reward_threshold=reward_threshold,
            recent_sample_prob=recent_sample_prob,
        )

        # --- World-model parameter group ---
        self.worldModelParameters = (
            list(self.recurrentModel.parameters())
            + list(self.posteriorNet.parameters())
            + list(self.priorNet.parameters())
            + list(self.sparseRewardPredictor.parameters())
            + list(self.standardRewardPredictor.parameters())
            + list(self.image_encoder.parameters())
            + list(self.teamitem_encoder.parameters())
            + list(self.ltm_reward_encoder.parameters())
            + list(self.grid_encoder.parameters())
            + list(self.teamitemPredictor.parameters())
            + list(self.ltm_reward_predictor.parameters())
            + list(self.grid_predictor.parameters())
            + list(self.decoder.parameters())
        )

        # --- Optimizers ---
        self.worldModelOptimizer = torch.optim.Adam(self.worldModelParameters, lr=2e-4)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=4e-5)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.curiosityCriticOptimizer = torch.optim.Adam(self.curiosity_critic.parameters(), lr=1e-4)
        self.curiosityHeadOptimizer = torch.optim.Adam(
            self.curiosityPredictor.parameters(), lr=1e-4)

    # ------------------------------------------------------
    def sample_batch(self, batchSize, sequenceSize):
        return self._batch_to_device(self.buffer.sample(batchSize, sequenceSize))

    def _batch_to_device(self, cpu_batch):
        """Move a pinned CPU batch to the device (non_blocking) and apply dtype conversions."""
        if cpu_batch is None:
            return None
        out = {k: v.to(self.device, non_blocking=True) for k, v in cpu_batch.items()}
        out["observations"] = out["observations"].float() / 255.0
        out["ltm_rewards"] = out["ltm_rewards"].float()
        out["grids"] = out["grids"].float()
        return out

    def computeLambdaValues(self, rewards, values, continues, lambda_=0.95):
        returns = torch.zeros_like(rewards)
        bootstrap = values[:, -1]
        for i in reversed(range(rewards.shape[-1])):
            returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i + 1] + lambda_ * bootstrap)
            bootstrap = returns[:, i]
        return returns

    def _pmpo_loss(self, advantages, log_probabilities):
        adv = advantages.detach()
        pos = (adv >= 0).float()
        neg = 1.0 - pos
        pos_logprob = (log_probabilities * pos).sum() / pos.sum().clamp(min=1.0)
        neg_logprob = (log_probabilities * neg).sum() / neg.sum().clamp(min=1.0)
        return -self.pmpo_alpha * pos_logprob + (1.0 - self.pmpo_alpha) * neg_logprob

    def _ltm_loss(self, pred_logits, target, pos_weight):
        """Weighted BCE + sparsity penalty for sparse whole-game multi-label targets."""
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
        sparsity = (torch.sigmoid(pred_logits) * (1.0 - target)).mean()
        return bce + self.ltm_sparsity_weight * sparsity

    # ------------------------------------------------------

    # ------------------------------------------------------
    def _encode_components(self, obs, ltm_reward, grid, team, item):
        """Encode all observation components (each input already on device)."""
        teamitem = torch.cat((team / 100.0, item / 10.0), dim=-1)
        return (
            self.image_encoder(obs),
            self.teamitem_encoder(teamitem),
            self.ltm_reward_encoder(ltm_reward),
            self.grid_encoder(grid),
        )

    # ------------------------------------------------------
    def TrainWorldModel(self, batch_data, compute_metrics=True):
        self.worldModelOptimizer.zero_grad(set_to_none=True)

        #batch_data: dict of tensors, each [B, T, ...]
        B, T = self.number_of_sequences, self.steps_per_sequence
        obs_flat = batch_data["observations"].flatten(0, 1)
        ltm_reward_flat = batch_data["ltm_rewards"].flatten(0, 1)
        grid_flat = batch_data["grids"].flatten(0, 1)
        team_flat = batch_data["team_levels"].flatten(0, 1)
        item_flat = batch_data["item_counts"].flatten(0, 1)

        # Encoding all observation components (image, team+item, long-term reward, grid)
        enc_img, enc_teamitem, enc_ltm_reward, enc_grid = self._encode_components(obs_flat, ltm_reward_flat, grid_flat, team_flat, item_flat)
        encoder_features = torch.cat((enc_img, enc_teamitem, enc_ltm_reward, enc_grid), dim=-1)
        posterior_flat, posterior_logits_flat = self.posteriorNet(encoder_features)
        posteriors_all = posterior_flat.view(B, T, self.latent_dim)
        posteriors_logits_all = posterior_logits_flat.view(B, T, self.rows, self.cols)

        # Recurrent rollout
        input_tokens = posteriors_all[:, :T - 1]
        recurrent_states, kv_context = self.recurrentModel.forward_sequence(
            input_tokens, batch_data["actions"][:, :T - 1], return_kv=True)

        prior_sample_flat, prior_logits_flat = self.priorNet(
            recurrent_states.reshape(-1, self.recurrent_dim))
        priors = prior_sample_flat.view(B, T - 1, self.latent_dim)
        priors_logits = prior_logits_flat.view(B, T - 1, self.rows, self.cols)

        posteriors = posteriors_all[:, 1:]
        posteriors_logits = posteriors_logits_all[:, 1:]
        full_states = torch.cat((recurrent_states, posteriors), -1)
        full_states_prior = torch.cat((recurrent_states, priors), -1)

        # reward calculation
        def _reward_loss(logits, target_scalar):
            with torch.no_grad():
                target_two_hot = self.two_hot.encode(target_scalar.squeeze(-1))
            return -torch.mean(torch.sum(
                target_two_hot * torch.log_softmax(logits, dim=-1), dim=-1))

        sparse_logits = self.sparseRewardPredictor(full_states)      # [B, T-1, num_bins]
        standard_logits = self.standardRewardPredictor(full_states)
        sparse_reward_loss = _reward_loss(sparse_logits, batch_data["sparse_rewards"][:, :-1])
        standard_reward_loss = _reward_loss(standard_logits, batch_data["standard_rewards"][:, :-1])
        reward_loss = sparse_reward_loss + standard_reward_loss

        # Combined team+item prediction (single decoder).
        pred_teamitem = self.teamitemPredictor(full_states)
        target_teamitem = torch.cat((batch_data["team_levels"][:, 1:] / 100.0, batch_data["item_counts"][:, 1:] / 10.0), dim=-1)
        teamitem_loss = F.mse_loss(pred_teamitem, target_teamitem)

        # Long memory loss.
        pred_ltm_reward = self.ltm_reward_predictor(full_states)
        ltm_reward_loss = self._ltm_loss(pred_ltm_reward, batch_data["ltm_rewards"][:, 1:], self.ltm_reward_pos_weight)

        # Local explored grid reconstruction
        target_grid = batch_data["grids"][:, 1:]
        pred_grid = self.grid_predictor(full_states)
        grid_bce = F.binary_cross_entropy_with_logits(pred_grid, target_grid, reduction='none')
        grid_cell_weight = 1.0 + (self.grid_zero_weight - 1.0) * (1.0 - target_grid)
        grid_loss = (grid_bce * grid_cell_weight).sum() / grid_cell_weight.sum()

        # KL loss.
        prior_distribution = Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1)
        prior_distribution_SG = Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1)
        posterior_distribution = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1)
        posterior_distribution_SG = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1)
        prior_loss = kl_divergence(posterior_distribution_SG, prior_distribution)
        posterior_loss = kl_divergence(posterior_distribution, prior_distribution_SG)
        freeNats = torch.full_like(prior_loss, 1)
        kl_loss = (1 * torch.maximum(prior_loss, freeNats) + 0.1 * torch.maximum(posterior_loss, freeNats)).mean()

        # --- Latent-Space Value-Alignment Regularization (Var) ---
        if self.total_num_updates > self.var_warmup_steps:
            value_logits_post = self.critic(full_states)         # V(v_t | s_t)
            value_logits_prior = self.critic(full_states_prior)  # V(v~_t | s~_t)
            logp_post = torch.log_softmax(value_logits_post, dim=-1)
            logp_prior = torch.log_softmax(value_logits_prior, dim=-1)
            p_post = logp_post.exp()
            # KL[ sg(V(v_t|s_t)) || V(v~_t|s~_t) ] — trains the prior (dynamics) branch.
            var_dyn = torch.sum(p_post.detach() * (logp_post.detach() - logp_prior), dim=-1)
            # KL[ V(v_t|s_t) || sg(V(v~_t|s~_t)) ] — trains the posterior (encoder) branch.
            var_reg = torch.sum(p_post * (logp_post - logp_prior.detach()), dim=-1)
            var_loss = (self.var_beta_dyn * var_dyn + self.var_beta_reg * var_reg).mean()
            beta_var = 1.0 / torch.clamp(prior_loss.mean().detach(), min=1.0)
            var_term = beta_var * var_loss
        else:
            var_loss = torch.zeros((), device=self.device)
            beta_var = torch.zeros((), device=self.device)
            var_term = torch.zeros((), device=self.device)

        # --- Image reconstruction loss (main world-model signal) ---
        target_imgs = batch_data["observations"][:, 1:].flatten(0, 1)
        recon_imgs = self.decoder(full_states.reshape(-1, self.concatenated_dim))
        recon_loss = 0.5 * ((recon_imgs - target_imgs) ** 2).flatten(1).sum(-1).mean()

        reward_loss = reward_loss*100
        teamitem_loss = teamitem_loss*100
        ltm_reward_loss = ltm_reward_loss*100
        grid_loss = grid_loss*100
        var_term = var_term*1000
        kl_loss = kl_loss*10
        # World Model Loss
        world_model_loss = (kl_loss + var_term
                            + reward_loss
                            + teamitem_loss
                            + ltm_reward_loss
                            + grid_loss
                            + recon_loss)

        # Backward pass, gradient clipping, and optimization step.
        world_model_loss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, 10.0, norm_type=2)
        self.worldModelOptimizer.step()

        # --- Curiosity head training on real states ---
        prior_states_flat = full_states_prior.detach().view(-1, self.concatenated_dim)
        curiosity_target_scalar = batch_data["curiosities"][:, :-1].squeeze(-1).reshape(-1)
        with torch.no_grad():
            target_curiosity = self.two_hot.encode(curiosity_target_scalar)
        self.curiosityHeadOptimizer.zero_grad(set_to_none=True)
        pred_curiosity_logits = self.curiosityPredictor(prior_states_flat)
        per_sample_ce = -torch.sum(
            target_curiosity * torch.log_softmax(pred_curiosity_logits, dim=-1), dim=-1)
        # Up-weight non-zero-curiosity samples: punish the head more for missing them.
        curiosity_weights = torch.ones_like(curiosity_target_scalar)
        curiosity_weights[curiosity_target_scalar > 0.0] = self.curiosity_pos_weight
        curiosity_loss = (per_sample_ce * curiosity_weights).sum() / curiosity_weights.sum()
        curiosity_loss.backward()
        curiosity_grad_norm = nn.utils.clip_grad_norm_(
            self.curiosityPredictor.parameters(), 10, norm_type=2)
        if torch.isfinite(curiosity_grad_norm):
            self.curiosityHeadOptimizer.step()
        else:
            print(f"[NaN guard] Skipped curiosity head step (grad norm: {curiosity_grad_norm.item()})")

        metrics = {}
        if compute_metrics:
            metrics = {
                "world_model_loss": world_model_loss.item(),
                "reward_loss": reward_loss.item(),
                "sparse_reward_loss": sparse_reward_loss.item(),
                "standard_reward_loss": standard_reward_loss.item(),
                "kl_loss": kl_loss.item(),
                "var_loss": var_term.item(),
                "beta_var": float(beta_var),
                "teamitem_loss": teamitem_loss.item(),
                "ltm_reward_loss": ltm_reward_loss.item(),
                "grid_loss": grid_loss.item(),
                "curiosity_loss": curiosity_loss.item(),
                "reconstruction_loss": recon_loss.item(),
            }
        return full_states.view(-1, self.concatenated_dim).detach(), kv_context, metrics

    # -----------------------------------------------------
    @staticmethod
    def _newly_activated_mask(logits, threshold=0.5):
        """Return a float mask [N, T] that is 1.0 at the first imagined step where any
        LTM bit turns ON (sigmoid > threshold) after being OFF at all earlier steps."""
        on = (torch.sigmoid(logits) > threshold)                       # [N, T, D] bool
        prev_on = torch.cumsum(on.float(), dim=1) - on.float()         # # of ON steps before t
        newly_on = on & (prev_on < 0.5)                                # on now, never on before
        return newly_on.any(dim=-1).float()                            # [N, T]

    def Dream(self, full_state, horizon=15,
              compute_metrics=True, kv_context=None):
        self.actorOptimizer.zero_grad(set_to_none=True)
        self.criticOptimizer.zero_grad(set_to_none=True)
        self.curiosityCriticOptimizer.zero_grad(set_to_none=True)

        all_states = full_state.detach()
        N = all_states.shape[0]

        Tr = self.steps_per_sequence - 1
        layout_ok = (N == self.number_of_sequences * Tr)

        # --- K dream starts per sequence (K = self.dreams_per_sequence)
        K = self.dreams_per_sequence
        if layout_ok:
            num_dreams = self.number_of_sequences * K
            t0 = torch.randint(0, Tr, (num_dreams,), device=all_states.device)
            seq_ids = torch.arange(self.number_of_sequences, device=all_states.device).repeat_interleave(K)
            base_idx = seq_ids * Tr + t0
        else:
            num_dreams = min(self.number_of_sequences * K, N)
            base_idx = torch.randint(0, N, (num_dreams,), device=all_states.device)

        sel_idx = base_idx
        start_states = all_states[sel_idx]


        # --- TSSM context: each dream start (b, t) gets its real replay history
        if layout_ok and kv_context is not None:
            cache = self.recurrentModel.make_cache(num_dreams, Tr + horizon, start_states.device)
            cache.prefill(kv_context, sel_idx // Tr, lengths=(sel_idx % Tr) + 1)
        else:
            cache = self.recurrentModel.make_cache(num_dreams, horizon + 1, start_states.device)

        # --- Imagination rollout (transformer decodes through the KV cache) ---
        full_states = [start_states]
        log_probabilities, entropies, actions_stack = [], [], []
        curr_state = start_states
        recurrent_state, latent_state = torch.split(curr_state, [self.recurrent_dim, self.latent_dim], -1)
        for _ in range(horizon):
            action, logprob, entropy = self.actor(curr_state)
            with torch.no_grad():
                recurrent_state = self.recurrentModel.forward_step(latent_state, action, cache)
                # Keep a diverging rollout from poisoning the actor / critics / reward heads.
                recurrent_state = torch.nan_to_num(recurrent_state, nan=0.0, posinf=1e4, neginf=-1e4)
                latent_state, _ = self.priorNet(recurrent_state, unimix=False)
            curr_state = torch.cat((recurrent_state, latent_state), -1)
            full_states.append(curr_state)
            log_probabilities.append(logprob)
            entropies.append(entropy)
            actions_stack.append(action)

        full_states = torch.stack(full_states, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        entropies = torch.stack(entropies, dim=1)
        actions_stack = torch.stack(actions_stack, dim=1)

        # --- Predicted rewards (single head each) and curiosity ---
        with torch.no_grad():
            imagined_steps = full_states[:, 1:]
            predicted_sparse = self.two_hot.decode(self.sparseRewardPredictor(imagined_steps)).squeeze(-1)
            predicted_standard = self.two_hot.decode(self.standardRewardPredictor(imagined_steps)).squeeze(-1)
            predicted_curiosity = self.two_hot.decode(self.curiosityPredictor(imagined_steps)).squeeze(-1)

            # --- Anti-duplication gate  --
            reward_ltm_logits = self.ltm_reward_predictor(imagined_steps)  # [N, T, D]
            predicted_sparse = predicted_sparse  #* self._newly_activated_mask(reward_ltm_logits, self.ltm_gate_threshold)
            predicted_rewards = predicted_sparse + predicted_standard
            grid_logits = self.grid_predictor(imagined_steps)             # [N, T, GRID_DIM]
            center_explored_prob = torch.sigmoid(grid_logits[..., GRID_CENTER_INDEX])
            tile_gate = 1.0 - center_explored_prob
            predicted_curiosity = predicted_curiosity #* tile_gate

        # --- Critic values (two-hot) ---
        imagined_states = full_states.detach()
        critic_logits = self.critic(imagined_states)
        curiosity_critic_logits = self.curiosity_critic(imagined_states)
        online_values = self.two_hot.decode(critic_logits.detach()).squeeze(-1)
        online_curiosity_values = self.two_hot.decode(curiosity_critic_logits.detach()).squeeze(-1)

        # --- Lambda returns ---
        with torch.no_grad():
            continues = torch.full_like(predicted_rewards, self.continue_discount)
            lambda_values = self.computeLambdaValues(predicted_rewards, online_values, continues)
            curiosity_lambda_values = self.computeLambdaValues(predicted_curiosity, online_curiosity_values, continues)

        # --- Advantages (raw; PMPO only uses their sign, so no normalization) ---
        reward_advantages = lambda_values - online_values[:, :-1].detach()
        curiosity_advantages = curiosity_lambda_values - online_curiosity_values[:, :-1].detach()

        # --- Actor loss (PMPO) ---
        actor_loss = (self._pmpo_loss(reward_advantages, log_probabilities)
                      + self.curiosity_scale * self._pmpo_loss(curiosity_advantages, log_probabilities)
                      - self.entropy_scale * entropies.mean())

        # --- Reward critic loss (CE to lambda returns) ---
        critic_logits_to_train = critic_logits[:, :-1]
        target_values_two_hot = self.two_hot.encode(lambda_values.detach())
        critic_loss = -torch.mean(torch.sum(target_values_two_hot * torch.log_softmax(critic_logits_to_train, dim=-1), dim=-1))

        # --- Curiosity critic loss (CE to lambda returns) ---
        curiosity_critic_logits_to_train = curiosity_critic_logits[:, :-1]
        target_curiosity_values_two_hot = self.two_hot.encode(curiosity_lambda_values.detach())
        curiosity_critic_loss = -torch.mean(torch.sum(
            target_curiosity_values_two_hot * torch.log_softmax(curiosity_critic_logits_to_train, dim=-1), dim=-1))

        # --- Optimization ---
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, norm_type=2)
        self.criticOptimizer.step()

        curiosity_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.curiosity_critic.parameters(), 1.0, norm_type=2)
        self.curiosityCriticOptimizer.step()

        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0, norm_type=2)
        self.actorOptimizer.step()

        metrics = {}
        if compute_metrics:
            metrics = {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "curiosity_critic_loss": curiosity_critic_loss.item(),
                "entropies": entropies.mean().item(),
                "log_probabilities": log_probabilities.mean().item(),
                "reward_advantages": reward_advantages.mean().item(),
                "advantages": reward_advantages.mean().item(),  # alias for policy.py logging
                "curiosity_advantages": curiosity_advantages.mean().item(),
                "advantage_pos_fraction": (reward_advantages >= 0).float().mean().item(),
                "critic_values": online_values.mean().item(),
                "curiosity_critic_values": online_curiosity_values.mean().item(),
                "dream_mean_curiosity": predicted_curiosity.mean().item(),
            }

        # --- Combined advantage, scaled exactly like the actor loss mixes the two PMPO terms ---
        combined_advantages = (reward_advantages + self.curiosity_scale * curiosity_advantages)  # [B, horizon]

        # --- Package representative trajectories for visualization ---
        def pack(idx, metric, label):
            return (metric,
                    full_states[idx].detach().cpu(),
                    predicted_rewards[idx].detach().cpu(),
                    lambda_values[idx].detach().cpu(),
                    actions_stack[idx].detach().cpu(),
                    reward_advantages[idx].detach().cpu(),
                    curiosity_advantages[idx].detach().cpu(),
                    combined_advantages[idx].detach().cpu(),
                    label)

        trajectory_advantages = combined_advantages.sum(dim=1)
        best_idx = torch.argmax(trajectory_advantages).item()
        rand_idx = torch.randint(0, trajectory_advantages.shape[0], (1,)).item()

        best_dream_data = pack(best_idx, trajectory_advantages[best_idx].item(), "MaxAdvantage")
        rand_dream_data = pack(rand_idx, trajectory_advantages[rand_idx].item(), "Random")
        return metrics, best_dream_data, rand_dream_data

    # ------------------------------------------------------
    @torch.no_grad()
    def Play_the_game(self, number_of_episodes_per_env=1, epsilon=0.05):
        num_envs = len(self.envs)
        episodes_completed = [0] * num_envs
        scores = []
        curiosity_scores = []
        current_rewards = [0.0] * num_envs
        current_curiosities = [0.0] * num_envs
        local_buffers = [[] for _ in range(num_envs)]
        maps_visited = [set() for _ in range(num_envs)]

        recurrent_state = torch.zeros((num_envs, self.recurrent_dim), device=self.device)
        context = self.recurrentModel.make_cache(num_envs, self.context_length, self.device)

        observations, ltm_rewards, grids, team_levels, item_counts = [], [], [], [], []
        for obs, info in self.envs.reset():  # all emulators reset in parallel
            observations.append(obs)
            ltm_rewards.append(np.array(info["ltm_reward"], dtype=np.float32))
            grids.append(np.array(info["grid"], dtype=np.float32))
            team_levels.append(np.array(info["team_levels"], dtype=np.float32))
            item_counts.append(np.array(info["item_counts"], dtype=np.float32))

        while min(episodes_completed) < number_of_episodes_per_env:
            obs_tensor = (torch.from_numpy(np.array(observations)).float() / 255.0).to(self.device)
            ltm_reward_tensor = torch.from_numpy(np.array(ltm_rewards)).float().to(self.device)
            grid_tensor = torch.from_numpy(np.array(grids)).float().to(self.device)
            team_tensor = torch.from_numpy(np.array(team_levels)).float().to(self.device)
            item_tensor = torch.from_numpy(np.array(item_counts)).float().to(self.device)

            enc_img, enc_teamitem, enc_ltm_reward, enc_grid = self._encode_components(
                obs_tensor, ltm_reward_tensor, grid_tensor, team_tensor, item_tensor)

            posterior_input = torch.cat((enc_img, enc_teamitem,
                                         enc_ltm_reward, enc_grid), -1)
            latent_state, _ = self.posteriorNet(posterior_input)
            action_onehot, _, _ = self.actor(torch.cat((recurrent_state, latent_state), -1))

            if epsilon > 0.0:
                override = torch.rand(num_envs, device=self.device) < epsilon
                if override.any():
                    rand_actions = torch.randint(0, self.action_dim, (int(override.sum().item()),), device=self.device)
                    action_onehot[override] = F.one_hot(rand_actions, self.action_dim).float()

            action = action_onehot
            # Append the executed token u_t=(z_t, a_t); the transformer returns h_{t+1}.
            recurrent_state = self.recurrentModel.forward_step(latent_state, action, context)
            action_idxs = torch.argmax(action, dim=-1).cpu().numpy()
            actions_for_buffer = action.cpu().numpy().astype(np.float32)

            # Step every still-active emulator at once (they run concurrently).
            active = [episodes_completed[i] < number_of_episodes_per_env for i in range(num_envs)]
            results = self.envs.step(action_idxs, active=active)

            envs_to_reset = []
            for i in range(num_envs):
                if not active[i]:
                    continue
                next_observation, reward, terminated, truncated, next_info = results[i]
                done = terminated or truncated
                for msg in next_info.get("logs", ()):
                    print(f"    [Env {i+1}] {msg}")
                self.total_num_steps += 1
                current_rewards[i] += reward
                current_map = next_info["coord"][0]
                if current_map not in maps_visited[i] and current_map in MAP_NAMES:
                    print(f"    [Env {i+1}] \033[1;96mEntered {MAP_NAMES[current_map]} for the first time\033[0m")
                maps_visited[i].add(current_map)
                sparse_reward = next_info.get("sparse_reward", 0.0)
                standard_reward = next_info.get("standard_reward", 0.0)
                curiosity = next_info.get("curiosity", 0.0)
                tier_event = next_info.get("tier_event", 0.0)
                current_curiosities[i] += curiosity

                local_buffers[i].append((
                    observations[i].copy(), ltm_rewards[i].copy(),
                    grids[i].copy(), item_counts[i].copy(), team_levels[i].copy(),
                    actions_for_buffer[i].copy(), sparse_reward, standard_reward,
                    curiosity, tier_event))

                observations[i] = next_observation
                ltm_rewards[i] = np.array(next_info["ltm_reward"], dtype=np.float32)
                grids[i] = np.array(next_info["grid"], dtype=np.float32)
                team_levels[i] = np.array(next_info["team_levels"], dtype=np.float32)
                item_counts[i] = np.array(next_info["item_counts"], dtype=np.float32)

                if done:
                    print(f"    [Env {i+1}] Episode done | Reward: {current_rewards[i]:.2f} | "
                          f"Curiosity: {current_curiosities[i]:.3f} | "
                          f"Unique maps visited: {len(maps_visited[i])} {sorted(maps_visited[i])}")
                    maps_visited[i] = set()
                    for transition in local_buffers[i]:
                        self.buffer.add(*transition)
                    local_buffers[i].clear()
                    self.buffer.end_episode()
                    scores.append(current_rewards[i])
                    curiosity_scores.append(current_curiosities[i])
                    self.total_num_episodes += 1
                    episodes_completed[i] += 1

                    if episodes_completed[i] < number_of_episodes_per_env:
                        envs_to_reset.append(i)
                        current_rewards[i] = 0.0
                        current_curiosities[i] = 0.0
                        recurrent_state[i] = torch.zeros(self.recurrent_dim, device=self.device)
                        context.reset_rows(i)  # clear the transformer's token history

            # Reset any finished envs (deferred out of the result loop; rare vs. steps).
            for i in envs_to_reset:
                next_obs, next_info = self.envs.reset_one(i)
                observations[i] = next_obs
                ltm_rewards[i] = np.array(next_info["ltm_reward"], dtype=np.float32)
                grids[i] = np.array(next_info["grid"], dtype=np.float32)
                team_levels[i] = np.array(next_info["team_levels"], dtype=np.float32)
                item_counts[i] = np.array(next_info["item_counts"], dtype=np.float32)


        for i in range(num_envs):
            if local_buffers[i]:
                for transition in local_buffers[i]:
                    self.buffer.add(*transition)
                local_buffers[i].clear()
                self.buffer.end_episode()

        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        avg_curiosity = round(sum(curiosity_scores) / len(curiosity_scores), 4) if curiosity_scores else 0.0
        return avg_score, avg_curiosity

    # ------------------------------------------------------
    # CHECKPOINTING 
    # ------------------------------------------------------
    _CHECKPOINT_MODULES = [
        'recurrentModel', 'posteriorNet', 'priorNet', 'sparseRewardPredictor', 'standardRewardPredictor',
        'curiosityPredictor',
        'image_encoder', 'teamitem_encoder', 'ltm_reward_encoder', 'grid_encoder',
        'teamitemPredictor', 'ltm_reward_predictor', 'grid_predictor',
        'actor', 'critic', 'curiosity_critic',
        'decoder',
    ]
    _CHECKPOINT_OPTIMIZERS = [
        'worldModelOptimizer', 'actorOptimizer', 'criticOptimizer',
        'curiosityCriticOptimizer', 'curiosityHeadOptimizer',
    ]

    def saveCheckpoints(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        names = self._CHECKPOINT_MODULES + self._CHECKPOINT_OPTIMIZERS
        checkpoint = {name: getattr(self, name).state_dict() for name in names}
        checkpoint.update({
            'total_num_episodes': self.total_num_episodes,
            'total_num_steps': self.total_num_steps,
            'total_num_updates': self.total_num_updates,
        })
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        buffer_path = os.path.join(directory, "replay_buffer.buffer") if directory else "replay_buffer.buffer"
        print("Saving replay buffer...")
        self.buffer.save(buffer_path)
        return 0

    def loadCheckpoints(self, path=None):
        if path is None:
            checkpoint_files = glob.glob("checkpoints/pokemon_model_R*_G*.pt")
            if not checkpoint_files:
                path = 'model.pt'
            else:
                try:
                    path = max(checkpoint_files, key=lambda x: int(x.split('_G')[-1].split('.pt')[0]))
                except Exception:
                    path = max(checkpoint_files, key=os.path.getctime)

        if not os.path.exists(path):
            print(f"No checkpoint found at {path}, starting from scratch.")
            return 0

        checkpoint = torch.load(path, map_location=self.device)
        # worldModelOptimizer is intentionally not reloaded (kept fresh, matching prior behavior).
        skip_optimizers = {'worldModelOptimizer'}
        for name in self._CHECKPOINT_MODULES + self._CHECKPOINT_OPTIMIZERS:
            if name in skip_optimizers or name not in checkpoint:
                if name not in checkpoint:
                    print(f"[load] '{name}' not in checkpoint -> fresh init.")
                continue
            try:
                getattr(self, name).load_state_dict(checkpoint[name])
            except Exception as e:
                print(f"[load] '{name}' shape/key mismatch -> fresh init ({e}).")

        self.total_num_episodes = checkpoint.get('total_num_episodes', 0)
        self.total_num_steps = checkpoint.get('total_num_steps', 0)
        self.total_num_updates = checkpoint.get('total_num_updates', 0)

        directory = os.path.dirname(path)
        buffer_path = os.path.join(directory, "replay_buffer.buffer") if directory else "replay_buffer.buffer"
        if os.path.exists(buffer_path):
            print("Loading replay buffer...")
            self.buffer.load(buffer_path)
        return 0

    # ------------------------------------------------------
    # DREAM VISUALIZATION (decoded frames + per-step reward/curiosity advantages)
    # ------------------------------------------------------
    @torch.no_grad()
    def visualize_single_dream(self, best_states, best_rewards, best_values, best_actions,
                               reward_advantages=None, curiosity_advantages=None,
                               combined_advantages=None, title_prefix="Dream",
                               label=None, max_advantage=None, pdf=None):
        best_states_device = best_states.to(self.device)
        grid_logits = self.grid_predictor(best_states_device)
        center_explored_prob = torch.sigmoid(grid_logits[..., GRID_CENTER_INDEX])
        tile_gate = 1.0 - center_explored_prob  # soft gate (matches Dream)
        curiosities = (
            self.two_hot.decode(self.curiosityPredictor(best_states_device)).squeeze(-1) * tile_gate
        ).cpu()

        decoded_imgs = self.decoder(best_states_device).clamp(0.0, 1.0).cpu()  # [horizon, 3, 64, 64]

        horizon = best_states.shape[0]
        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
        action_icons = {"UP": "▲ UP", "DOWN": "▼ DN", "LEFT": "◀ LT", "RIGHT": "▶ RT", "A": "A", "B": "B"}

        # Taller info row + larger figure so the per-step numbers are easy to read.
        fig, axes = plt.subplots(2, horizon, figsize=(horizon * 2.7, 7.5), facecolor='#0d1117', dpi=120,
                                 gridspec_kw={'height_ratios': [1.4, 1.3]})
        # Title: dream type + its (max) combined advantage.
        dream_type = label if label is not None else "Dream"
        adv_value = max_advantage
        if adv_value is None and combined_advantages is not None:
            adv_value = combined_advantages.sum().item()
        title_txt = f'{dream_type} Dream'
        if adv_value is not None:
            title_txt += f'  —  Advantage = {adv_value:+.3f}'
        fig.suptitle(title_txt, color='#58a6ff', fontsize=20, fontweight='bold', y=0.99)
        if horizon == 1:
            axes = np.expand_dims(axes, axis=1)

        def _col(v):
            return '#3fb950' if v > 0 else ('#ff7b72' if v < 0 else '#8b949e')

        for i in range(horizon):
            ax_img, ax_info = axes[0, i], axes[1, i]
            ax_img.imshow(decoded_imgs[i].permute(1, 2, 0).numpy())
            ax_img.axis('off')
            for spine in ax_img.spines.values():
                spine.set_visible(True); spine.set_edgecolor('#30363d'); spine.set_linewidth(1.0)

            ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1); ax_info.axis('off'); ax_info.set_facecolor('#161b22')
            for spine in ax_info.spines.values():
                spine.set_visible(True); spine.set_edgecolor('#30363d'); spine.set_linewidth(0.5)

            if i < horizon - 1:
                step_reward = best_rewards[i].item()
                step_cur = curiosities[i].item()
                action_idx = torch.argmax(best_actions[i]).item()
                action_str = action_names[action_idx] if action_idx < len(action_names) else str(action_idx)
                icon = action_icons.get(action_str, action_str)

                # Per step: action, reward, curiosity, combined (scaled) advantage.
                lines = [
                    (icon, '#58a6ff', 17),
                    (f'r: {step_reward:+.2f}', _col(step_reward), 15),
                    (f'c: {step_cur:+.4f}', '#d1f1a5', 15),
                ]
                if combined_advantages is not None:
                    a_comb = combined_advantages[i].item()
                    lines.append((f'A: {a_comb:+.3f}', _col(a_comb), 16))

                y = 0.90
                for txt, col, fsize in lines:
                    ax_info.text(0.5, y, txt, ha='center', va='center', fontsize=fsize,
                                 fontweight='bold', color=col,
                                 fontfamily='monospace', transform=ax_info.transAxes)
                    y -= 0.24
            else:
                ax_info.text(0.5, 0.72, 'END', ha='center', va='center', fontsize=18,
                             fontweight='bold', color='#8b949e', transform=ax_info.transAxes)
                ax_info.text(0.5, 0.40, f'c: {curiosities[i].item():+.4f}', ha='center', va='center',
                             fontsize=15, fontweight='bold', color='#d1f1a5',
                             fontfamily='monospace', transform=ax_info.transAxes)

        plt.subplots_adjust(top=0.90, bottom=0.04, hspace=0.12)
        # Save to the PdfPages handle if given, otherwise show interactively.
        if pdf is not None:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
        else:
            plt.show()
        plt.close(fig)
