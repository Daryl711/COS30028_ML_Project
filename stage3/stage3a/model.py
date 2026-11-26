"""R4: 2b with learnable prompts (guide.md Stage 3, PROPOSED_METHOD.md §3.3 and §4.5).

  v   = BioCLIP.encode_image(x)                512-d, frozen (kept outside this module)
  f   = N(v)                                   same image feature for both heads, as in 2b (no adapters)
  T_c = N(E_t([V^c_1..V^c_4] [θ_crop] leaf, {taxonomy}))              55 x 512
  T_d = N(E_t([V^d_1..V^d_4] leaf with [θ_disease], {taxonomy}))     175 x 512
  crop logits    = s * cos(Dropout(f), T_c)     s = BioCLIP's pretrained logit scale, fixed
  disease logits = s * cos(Dropout(f), T_d)
  L_kg = mean_k ||T_c[k] - T_c^0[k]||^2 + mean_k ||T_d[k] - T_d^0[k]||^2     (KgCoOp anchor)

E_t is BioCLIP's text encoder, frozen; gradients pass through it to the prompt tokens.
  - V (CoOp context, M = 4): one set per head, shared by all classes of that head. Initialised from the
    token embeddings of "a photo of a", the first four words of the taxonomy prompt.
  - θ (CSP primitive tokens): one vector per subword of each class name, in the label slot. Initialised
    from the name's own token embeddings. The rest of the taxonomy string stays fixed text. During
    training θ gets CSP's attribute dropout (0.3, training_scheme.md §6.1); eval and T^0 have none.
  - T^0 is the frozen embedding of the unchanged taxonomy prompts, i.e. exactly the 2b / R5 initial text
    classifier.

Both token sets are stored as init + delta with delta zero-initialised, so AdamW's weight decay pulls a
token toward its pretrained embedding rather than toward the zero vector. At epoch 0 every delta is zero,
so T_c = T_c^0 and T_d = T_d^0 and R4 starts exactly at 2b's epoch 0.

Unlike 2b and R5, T_c and T_d are not free parameters: every class vector is an output of the text encoder.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stage3.stage3b.dataset import _sp


class PromptLearner(nn.Module):
    """One head's prompts: CoOp context + CSP class-name tokens, encoded by the frozen BioCLIP text tower.

    `prefix` is the template text before the class name ("a photo of a" or "a photo of a leaf with");
    its first `n_ctx` tokens become the learnable context. The text tower is held by reference, not
    registered as a submodule, so it is never trained, moved or saved with this module.
    """

    def __init__(self, clip, tokenizer, prompts: list[str], names: list[str], prefix: str,
                 n_ctx: int, device: torch.device, attr_dropout: float = 0.3):
        super().__init__()
        if clip.text_pool_type != "argmax" or not isinstance(clip.text_projection, nn.Parameter):
            raise RuntimeError("expected an OpenAI-style CLIP text tower (argmax pooling, projection matrix)")
        self._clip = [clip]                              # plain list: keeps the tower out of the module tree

        ids = tokenizer(prompts)                         # [K, 77]
        if int((ids != 0).sum(1).max()) >= ids.shape[1]:
            raise ValueError("a prompt hit the 77-token limit")
        # The full 77-token context is kept (not cut at the longest prompt): with it the prompt path is
        # bit-identical to clip.encode_text at init, so epoch 0 equals 2b's exactly. Cutting the padding
        # is mathematically the same (causal attention) but changes fp16 rounding.
        L = ids.shape[1]

        pre = tokenizer([prefix])[0]
        n_pre = int((pre != 0).sum()) - 2                # without [SOS] / [EOS]
        if n_pre < n_ctx:
            raise ValueError(f"prefix {prefix!r} has {n_pre} tokens, fewer than n_ctx={n_ctx}")
        if not torch.equal(ids[:, 1:1 + n_pre], pre[1:1 + n_pre].expand(len(prompts), -1)):
            raise ValueError(f"not every prompt starts with {prefix!r}")

        # CSP slots: the class name's subword tokens directly after the prefix
        rows, cols = [], []
        name_tok = tokenizer([_sp(n) for n in names])
        for k in range(len(names)):
            n_tok = int((name_tok[k] != 0).sum()) - 2
            start = 1 + n_pre
            if not torch.equal(ids[k, start:start + n_tok], name_tok[k, 1:1 + n_tok]):
                raise ValueError(f"class name {names[k]!r} is not at position {start} of its prompt")
            rows += [k] * n_tok
            cols += list(range(start, start + n_tok))

        ids = ids.to(device)
        with torch.no_grad():
            base = clip.token_embedding(ids).float()     # [K, L, D] pretrained token embeddings
        dim = base.shape[-1]
        self.n_ctx = n_ctx
        self.register_buffer("ids", ids, persistent=False)
        self.register_buffer("base", base, persistent=False)
        self.register_buffer("name_rows", torch.tensor(rows, device=device), persistent=False)
        self.register_buffer("name_cols", torch.tensor(cols, device=device), persistent=False)
        self.register_buffer("attn_mask", clip.attn_mask[:L, :L].to(device), persistent=False)
        self.ctx_delta = nn.Parameter(torch.zeros(n_ctx, dim, device=device))           # CoOp V_1..V_M
        self.name_delta = nn.Parameter(torch.zeros(len(rows), dim, device=device))       # CSP θ
        # CSP attribute dropout (training_scheme.md §6.1: 0.3), on the class-name tokens θ, training only
        self.attr_drop = nn.Dropout(attr_dropout) if attr_dropout > 0 else nn.Identity()

    @property
    def num_name_tokens(self) -> int:
        return self.name_delta.shape[0]

    def token_embeddings(self) -> torch.Tensor:
        delta = torch.zeros_like(self.base)
        delta[:, 1:1 + self.n_ctx] = self.ctx_delta                   # shared by every class of this head
        emb = self.base + delta
        theta = self.base[self.name_rows, self.name_cols] + self.name_delta
        return emb.index_put((self.name_rows, self.name_cols), self.attr_drop(theta))

    def forward(self) -> torch.Tensor:
        """L2-normalised class text features [K, 512]. Follows open_clip CLIP._encode_text step by step;
        run it under the same autocast as the image encoder."""
        clip = self._clip[0]
        cast = clip.transformer.get_cast_dtype()
        L = self.ids.shape[1]
        x = self.token_embeddings().to(cast) + clip.positional_embedding[:L].to(cast)
        x = clip.transformer(x, attn_mask=self.attn_mask)
        x = clip.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), self.ids.argmax(dim=-1)]   # EOT token
        x = x @ clip.text_projection
        return F.normalize(x.float(), dim=-1)


class TextHeadNet(nn.Module):
    """Inference form of R4 (and of 2b): fixed class vectors, f = N(v) for both heads.

    The checkpoint's "model" entry is this module's state dict (Tc, Td, logit_scale), with Tc, Td the
    prompt learner's output for that epoch. Prediction needs no text encoder, so the Stage 3 evaluator
    runs on it unchanged.
    """

    def __init__(self, text_crop: torch.Tensor, text_disease: torch.Tensor, logit_scale: float,
                 head_dropout: float = 0.3):
        super().__init__()
        self.register_buffer("Tc", F.normalize(text_crop.float(), dim=-1))
        self.register_buffer("Td", F.normalize(text_disease.float(), dim=-1))
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))
        self.drop = nn.Dropout(head_dropout)

    def features(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        f = F.normalize(v.float(), dim=-1)
        return {"crop": f, "disease": f}

    def forward(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        f = F.normalize(self.drop(self.features(v)["crop"]), dim=-1)
        return {"crop": self.logit_scale * f @ self.Tc.T, "disease": self.logit_scale * f @ self.Td.T}


class R4Net(nn.Module):
    """Everything trainable in R4: the crop and disease prompt tokens. Input is the raw BioCLIP feature v."""

    def __init__(self, clip, tokenizer, prompts: dict[str, list[str]], crops: list[str], diseases: list[str],
                 logit_scale: float, head_dropout: float, n_ctx: int, device: torch.device,
                 attr_dropout: float = 0.3):
        super().__init__()
        self.crop_prompts = PromptLearner(clip, tokenizer, prompts["crop"], crops, "a photo of a", n_ctx,
                                          device, attr_dropout)
        self.disease_prompts = PromptLearner(clip, tokenizer, prompts["disease"], diseases,
                                             "a photo of a leaf with", n_ctx, device, attr_dropout)
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale), device=device))
        self.head_dropout = head_dropout
        self.drop = nn.Dropout(head_dropout)
        # KgCoOp anchors T^0: the prompts with every delta at zero and no dropout (= the unchanged taxonomy text)
        t0 = self._eval_text()
        self.register_buffer("T0c", t0["crop"].clone(), persistent=False)
        self.register_buffer("T0d", t0["disease"].clone(), persistent=False)

    @torch.no_grad()
    def _eval_text(self) -> dict[str, torch.Tensor]:
        """Text features in eval mode (no attribute dropout), restoring the previous mode."""
        was_training = self.training
        self.eval()
        dev = self.logit_scale.device
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            T = self.text()
        self.train(was_training)
        return T

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """One AdamW param group per trainable module (training_scheme.md §3)."""
        mods = (("crop_ctx", [self.crop_prompts.ctx_delta]), ("crop_name_tokens", [self.crop_prompts.name_delta]),
                ("disease_ctx", [self.disease_prompts.ctx_delta]),
                ("disease_name_tokens", [self.disease_prompts.name_delta]))
        return [{"name": n, "params": p, "lr": lr, "weight_decay": weight_decay} for n, p in mods]

    def text(self) -> dict[str, torch.Tensor]:
        return {"crop": self.crop_prompts(), "disease": self.disease_prompts()}

    def features(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        f = F.normalize(v.float(), dim=-1)
        return {"crop": f, "disease": f}

    def forward(self, v: torch.Tensor, T: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Logits for image features v, given this step's text features T (from `text()`)."""
        f = F.normalize(self.drop(self.features(v)["crop"]), dim=-1)
        return {"crop": self.logit_scale * f @ T["crop"].float().T,
                "disease": self.logit_scale * f @ T["disease"].float().T}

    def drift(self, T: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """mean_k ||T[k] - T^0[k]||^2 per head; L_kg is their sum."""
        return {"crop": (T["crop"].float() - self.T0c).pow(2).sum(-1).mean(),
                "disease": (T["disease"].float() - self.T0d).pow(2).sum(-1).mean()}

    def frozen(self) -> TextHeadNet:
        """Snapshot of the current classifier (eval mode, no dropout), for validation and the checkpoint."""
        T = self._eval_text()
        return TextHeadNet(T["crop"], T["disease"], self.logit_scale.item(),
                           self.head_dropout).to(self.logit_scale.device)
