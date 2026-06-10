#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bash /configs/patches/vllm-container-deps-cutlass-cu13.sh

python3 - <<'PY'
from pathlib import Path
import glob
import site
import sys


def candidate_roots() -> list[Path]:
    roots: list[Path] = []

    for getter in (site.getsitepackages,):
        try:
            roots.extend(Path(p) for p in getter())
        except Exception:
            pass

    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass

    roots.extend(Path(p) for p in sys.path if p)
    roots.extend(Path(p) for p in glob.glob("/usr/local/lib/python*/dist-packages"))
    roots.extend(Path(p) for p in glob.glob("/usr/local/lib/python*/site-packages"))
    return roots


def find_installed_file(rel: str) -> Path:
    seen: set[Path] = set()
    for root in candidate_roots():
        target = (root / rel).resolve()
        if target in seen:
            continue
        seen.add(target)
        if target.exists():
            return target
    raise RuntimeError(f"Could not find installed vLLM file: {rel}")


def replace_once(text: str, old: str, new: str, path: Path, desc: str) -> str:
    if old not in text:
        raise RuntimeError(f"{path}: expected block not found while patching {desc}")
    return text.replace(old, new, 1)


kernel_warmup_path = find_installed_file(
    "vllm/model_executor/warmup/kernel_warmup.py"
)
model_path = find_installed_file("vllm/models/deepseek_v4/nvidia/model.py")

kernel_text = kernel_warmup_path.read_text()
if "model_specific_warmup = getattr(model, \"warmup_model_specific_kernels\", None)" in kernel_text:
    print(f"[megamoe-warmup-hotfix] {kernel_warmup_path}: already patched")
else:
    kernel_text = replace_once(
        kernel_text,
        '''def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
''',
        '''def kernel_warmup(worker: "Worker"):
    model = worker.get_model()

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
''',
        kernel_warmup_path,
        "kernel_warmup model lookup",
    )
    kernel_text = replace_once(
        kernel_text,
        '''    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    enable_flashinfer_autotune = (
''',
        '''    if do_deep_gemm_warmup:
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    model_specific_warmup = getattr(model, "warmup_model_specific_kernels", None)
    if callable(model_specific_warmup):
        model_specific_warmup(worker.scheduler_config.max_num_batched_tokens)

    enable_flashinfer_autotune = (
''',
        kernel_warmup_path,
        "kernel_warmup model-specific hook",
    )
    kernel_warmup_path.write_text(kernel_text)
    print(f"[megamoe-warmup-hotfix] patched {kernel_warmup_path}")

model_text = model_path.read_text()
if "Compile the direct DeepGEMM MegaMoE path before serving traffic." in model_text:
    print(f"[megamoe-warmup-hotfix] {model_path}: already patched")
else:
    model_text = replace_once(
        model_text,
        '''        return y

    def _run_mega_moe(
''',
        '''        return y

    @torch.inference_mode()
    def warmup_kernels(self, max_tokens: int) -> None:
        """Compile the direct DeepGEMM MegaMoE path before serving traffic."""
        self.finalize_weights()
        symm_buffer = self.get_symm_buffer()
        device = torch.device("cuda", torch.cuda.current_device())

        max_tokens = max(1, min(max_tokens, self.max_num_tokens))
        token_counts = sorted({1, max_tokens})

        ep_size = max(1, self.num_experts // self.num_local_experts)
        ep_rank = self.experts_start_idx // self.num_local_experts
        topk_offsets = torch.arange(self.top_k, device=device, dtype=torch.int64)

        for num_tokens in token_counts:
            hidden_states = torch.ones(
                (num_tokens, self.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
            token_offsets = torch.arange(
                num_tokens, device=device, dtype=torch.int64
            )[:, None]
            topk_ranks = (ep_rank + token_offsets + topk_offsets[None, :]) % ep_size
            local_experts = (
                token_offsets * self.top_k + topk_offsets[None, :]
            ) % self.num_local_experts
            topk_ids = topk_ranks * self.num_local_experts + local_experts
            topk_weights = torch.full(
                (num_tokens, self.top_k),
                1.0 / self.top_k,
                device=device,
                dtype=torch.float32,
            )
            y = torch.empty_like(hidden_states, dtype=torch.bfloat16)

            prepare_megamoe_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                symm_buffer.x[:num_tokens],
                symm_buffer.x_sf[:num_tokens],
                symm_buffer.topk_idx[:num_tokens],
                symm_buffer.topk_weights[:num_tokens],
            )
            torch.accelerator.synchronize()
            get_ep_group().barrier()

            self._run_mega_moe(
                hidden_states,
                topk_weights,
                topk_ids,
                y,
                activation_clamp=None,
                fast_math=True,
            )
            torch.accelerator.synchronize()
            get_ep_group().barrier()

    def _run_mega_moe(
''',
        model_path,
        "DeepseekV4MegaMoEExperts warmup_kernels",
    )

    model_text = replace_once(
        model_text,
        '''    def finalize_mega_moe_weights(self) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.ffn.finalize_mega_moe_weights()


def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
''',
        '''    def finalize_mega_moe_weights(self) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.ffn.finalize_mega_moe_weights()

    def warmup_model_specific_kernels(self, max_tokens: int) -> None:
        if not self.use_mega_moe:
            return

        seen: set[tuple[int, int, int, int, int, int]] = set()
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            ffn = getattr(layer, "ffn", None)
            if ffn is None or not getattr(ffn, "use_mega_moe", False):
                continue
            experts = ffn.experts
            key = (
                experts.num_experts,
                experts.num_local_experts,
                experts.top_k,
                experts.hidden_size,
                experts.intermediate_size,
                experts.max_num_tokens,
            )
            if key in seen:
                continue
            seen.add(key)
            experts.warmup_kernels(max_tokens)


def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
''',
        model_path,
        "DeepseekV4Model warmup_model_specific_kernels",
    )

    model_text = replace_once(
        model_text,
        '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
''',
        '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        return loaded_params

    def warmup_model_specific_kernels(self, max_tokens: int) -> None:
        self.model.warmup_model_specific_kernels(max_tokens)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
''',
        model_path,
        "DeepseekV4ForCausalLM warmup_model_specific_kernels",
    )

    model_path.write_text(model_text)
    print(f"[megamoe-warmup-hotfix] patched {model_path}")

print("[megamoe-warmup-hotfix] complete")
PY
