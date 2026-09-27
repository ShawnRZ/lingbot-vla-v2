# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightweight LoRA for LingBot-VLA.

`LoRALinear` subclasses `nn.Linear` and keeps the original `weight` / `bias`
parameter names, so:
  - pretrained weights load into it unchanged,
  - model code that reads `xxx_proj.weight.dtype` keeps working,
  - the only extra state_dict keys are `<prefix>.lora_A`, `<prefix>.lora_B`
    and the persistent buffer `<prefix>.lora_scaling`, which
    `merge_lora_state_dict` folds back into `<prefix>.weight` on export.
"""

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


LORA_PARAM_PATTERN = "lora_"


class LoRALinear(nn.Linear):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        # Share the pretrained parameters instead of copying them.
        self.weight = base.weight
        self.bias = base.bias

        device, dtype = base.weight.device, base.weight.dtype
        self.lora_rank = rank
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer(
            "lora_scaling", torch.tensor(alpha / rank, device=device, dtype=torch.float32), persistent=True
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + lora_out * self.lora_scaling.to(lora_out.dtype)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, lora_rank={self.lora_rank}"


def inject_lora(
    model: nn.Module,
    target_module_patterns: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[str]:
    """Replace every nn.Linear whose FQN matches any regex in `target_module_patterns` (re.search) with LoRALinear."""
    patterns = [re.compile(p) for p in target_module_patterns]
    targets = [
        name
        for name, module in model.named_modules()
        if type(module) is nn.Linear and any(p.search(name) for p in patterns)
    ]
    for name in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), rank, alpha, dropout))
    return targets


def apply_lora_and_freeze(
    model: nn.Module,
    lora_target_modules: Iterable[str],
    trainable_patterns: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[str]:
    """
    Freeze the whole model, inject LoRA into `lora_target_modules`, then
    unfreeze LoRA params plus every param whose FQN matches `trainable_patterns` (re.search).
    """
    model.requires_grad_(False)
    targets = inject_lora(model, lora_target_modules, rank, alpha, dropout)
    trainable = [re.compile(p) for p in trainable_patterns]
    for name, param in model.named_parameters():
        if LORA_PARAM_PATTERN in name.rsplit(".", 1)[-1] or any(p.search(name) for p in trainable):
            param.requires_grad_(True)
    return targets


_MOE_SUBMODULES = ("experts", "shared_expert", "shared_expert_gate", "gate")


def _summary_key(name: str) -> str:
    parts = re.sub(r"\.\d+(?=\.|$)", ".N", name).split(".")
    if "qwenvl_with_expert" not in parts:
        return ".".join(parts[:2])  # e.g. model.state_proj / model.depth_align_head
    if "layers" in parts:
        # e.g. ...layers.N.self_attn / ...layers.N.mlp / ...layers.N.mlp.experts / ...layers.N.input_layernorm
        i = parts.index("layers")
        end = i + 3
        if parts[i + 2] == "mlp" and len(parts) > i + 4 and parts[i + 3] in _MOE_SUBMODULES:
            end += 1
        return ".".join(parts[:end])
    return ".".join(parts[:5])


def summarize_trainable_params(model: nn.Module) -> str:
    """Group params by decoder sub-module / top-level module and report trainable/total."""
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for name, param in model.named_parameters():
        key = _summary_key(name)
        if LORA_PARAM_PATTERN in name.rsplit(".", 1)[-1]:
            key += " [lora]"
        stats[key][1] += param.numel()
        if param.requires_grad:
            stats[key][0] += param.numel()

    total = sum(v[1] for v in stats.values())
    trainable = sum(v[0] for v in stats.values())
    lines = [f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M ({100 * trainable / max(total, 1):.2f}%)"]
    for key, (t, n) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        tag = "train" if t == n else ("frozen" if t == 0 else "partial")
        lines.append(f"  [{tag:>7}] {n / 1e6:10.2f}M  {key}")
    return "\n".join(lines)


@torch.no_grad()
def merge_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Fold `weight += lora_scaling * lora_B @ lora_A` and drop all LoRA keys, in place.
    A no-op for state dicts without LoRA keys.
    """
    suffix = ".lora_A"
    prefixes = [k[: -len(suffix)] for k in state_dict if k.endswith(suffix)]
    for prefix in prefixes:
        lora_a = state_dict.pop(f"{prefix}.lora_A")
        lora_b = state_dict.pop(f"{prefix}.lora_B")
        scaling = state_dict.pop(f"{prefix}.lora_scaling", None)
        if scaling is None:
            raise KeyError(f"Missing {prefix}.lora_scaling, cannot merge LoRA weights.")
        weight = state_dict[f"{prefix}.weight"]
        delta = (lora_b.float() @ lora_a.float()) * float(scaling)
        state_dict[f"{prefix}.weight"] = (weight.float() + delta.to(weight.device)).to(weight.dtype)
    return state_dict

