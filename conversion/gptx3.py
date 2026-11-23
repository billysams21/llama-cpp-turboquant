from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("GPTX3ForCausalLM")
class GPTX3Model(TextModel):
    """GPT-X3 (Axiomic Labs): hybrid GDN2 (gated delta net v2) + gated GQA, 3:1 layout"""
    model_arch = gguf.MODEL_ARCH.GPTX3

    # pre-tokenizer regex used by the LFM2(.5) tokenizer (identical to llama3)
    _LFM2_PRE_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    def get_vocab_base_pre(self, tokenizer) -> str:
        # GPT-X3 uses the LFM2.5 tokenizer with BOS removed from the post-processor,
        # which changes the checksum but not the pre-tokenizer itself.
        tokenizer_json = self.dir_model / "tokenizer.json"
        if tokenizer_json.is_file():
            with open(tokenizer_json, encoding="utf-8") as f:
                pre = json.load(f).get("pre_tokenizer") or {}
            for p in pre.get("pretokenizers", [pre]):
                if p.get("type") == "Split" and p.get("pattern", {}).get("Regex") == self._LFM2_PRE_REGEX:
                    return "lfm2"
        return super().get_vocab_base_pre(tokenizer)

    def set_vocab(self):
        self._set_vocab_gpt2()
        # the model is trained without BOS
        self.gguf_writer.add_add_bos_token(False)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams
        n_head = hp["num_attention_heads"]
        gdn_head_dim = hp["gdn_head_dim"]
        # the llama.cpp graph uses n_head for both GQA query heads and GDN2 heads
        assert hp["hidden_size"] == gdn_head_dim * n_head, "GDN2 heads must match attention heads"

        period = hp["gdn_layers_per_gqa"] + 1
        # n_head_kv == 0 marks GDN2 (recurrent) layers
        self.gguf_writer.add_head_count_kv(
            [hp["num_key_value_heads"] if (il + 1) % period == 0 else 0 for il in range(self.block_count)])
        self.gguf_writer.add_rope_dimension_count(hp["rotary_dim"])
        self.gguf_writer.add_ssm_conv_kernel(hp["conv_size"])
        self.gguf_writer.add_kda_head_dim(gdn_head_dim)
        self.gguf_writer.add_group_norm_eps(hp["gdn_norm_eps"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        MT = gguf.MODEL_TENSOR

        if name == "transformer.wte.weight":
            yield self.format_tensor_name(MT.TOKEN_EMBD), data_torch
            return
        if name == "transformer.ln_f.weight":
            yield self.format_tensor_name(MT.OUTPUT_NORM), data_torch
            return
        if name == "lm_head.weight":
            yield self.format_tensor_name(MT.OUTPUT), data_torch
            return

        assert bid is not None
        prefix = f"transformer.h.{bid}."
        if not name.startswith(prefix):
            raise ValueError(f"Can not map tensor {name!r}")
        sub = name[len(prefix):]

        mapping: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "ln_1.weight":                 (MT.ATTN_NORM,    ".weight"),
            "ln_2.weight":                 (MT.FFN_NORM,     ".weight"),
            "mlp.w_gate.weight":           (MT.FFN_GATE,     ".weight"),
            "mlp.w_up.weight":             (MT.FFN_UP,       ".weight"),
            "mlp.w_down.weight":           (MT.FFN_DOWN,     ".weight"),
            # gated GQA layers
            "attn.q_gate_proj.weight":     (MT.ATTN_Q,       ".weight"),
            "attn.k_proj.weight":          (MT.ATTN_K,       ".weight"),
            "attn.v_proj.weight":          (MT.ATTN_V,       ".weight"),
            "attn.o_proj.weight":          (MT.ATTN_OUT,     ".weight"),
            "attn.q_norm.weight":          (MT.ATTN_Q_NORM,  ".weight"),
            "attn.k_norm.weight":          (MT.ATTN_K_NORM,  ".weight"),
            # GDN2 layers
            "attn.kernel.q_proj.weight":   (MT.ATTN_Q,       ".weight"),
            "attn.kernel.k_proj.weight":   (MT.ATTN_K,       ".weight"),
            "attn.kernel.v_proj.weight":   (MT.ATTN_V,       ".weight"),
            "attn.kernel.o_proj.weight":   (MT.ATTN_OUT,     ".weight"),
            "attn.kernel.q_conv1d.weight": (MT.SSM_CONV1D_Q, ".weight"),
            "attn.kernel.k_conv1d.weight": (MT.SSM_CONV1D_K, ".weight"),
            "attn.kernel.v_conv1d.weight": (MT.SSM_CONV1D_V, ".weight"),
            "attn.kernel.f_proj.0.weight": (MT.SSM_F_A,      ".weight"),
            "attn.kernel.f_proj.1.weight": (MT.SSM_F_B,      ".weight"),
            "attn.kernel.g_proj.0.weight": (MT.SSM_G_A,      ".weight"),
            "attn.kernel.g_proj.1.weight": (MT.SSM_G_B,      ".weight"),
            "attn.kernel.g_proj.1.bias":   (MT.SSM_G_B,      ".bias"),
            "attn.kernel.b_proj.weight":   (MT.SSM_BETA,     ".weight"),
            "attn.kernel.w_proj.weight":   (MT.ATTN_GATE,    ".weight"),
            "attn.kernel.A_log":           (MT.SSM_A,        ""),
            "attn.kernel.dt_bias":         (MT.SSM_DT,       ".bias"),
            "attn.kernel.o_norm.weight":   (MT.SSM_NORM,     ".weight"),
        }
        if sub not in mapping:
            raise ValueError(f"Can not map tensor {name!r}")
        key, suffix = mapping[sub]

        if sub in ("attn.q_norm.weight", "attn.k_norm.weight"):
            # OffsetRMSNorm: y * (1 + w)
            data_torch = data_torch + 1.0
        elif sub == "attn.kernel.A_log":
            data_torch = -torch.exp(data_torch.float())
        elif sub.endswith("_conv1d.weight"):
            # [d_inner, 1, d_conv] -> [d_inner, d_conv] (ggml ne = [d_conv, d_inner])
            data_torch = data_torch.reshape(data_torch.shape[0], data_torch.shape[-1])

        yield self.format_tensor_name(key, bid, suffix), data_torch
