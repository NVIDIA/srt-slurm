"""
Free original DeepSeek V4 MoE expert weights after MegaMoE finalize.

Symptom (seen on GB200 decode, EP=8, VLLM_DEEPSEEK_V4_USE_MEGA_MOE=1):
    torch.OutOfMemoryError: Tried to allocate 1008.00 MiB.
    GPU 0 has a total capacity of 184.31 GiB of which 381.44 MiB is free.
    181.02 GiB allocated by PyTorch.
  Stack ends in deep_gemm/mega/__init__.py interleave():
    torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(...))

Root cause: DeepseekV4MegaMoEExperts.finalize_weights() builds
self._transformed_l1_weights / _transformed_l2_weights but does NOT release
the original self.w13_weight / w2_weight / *_weight_scale parameters. Both
copies stay resident on GPU through finalize iteration, and on EP=8 the
per-rank weight footprint (~125 GiB) plus this duplication leaves no
headroom for the per-layer interleave temporaries (~1 GiB peak).

Forward path verified (deepseek_v4.py: _run_mega_moe, ~line 538-547) only
reads self._transformed_l1_weights / _transformed_l2_weights. Original
w13_weight / w2_weight / *_weight_scale are dead after finalize.

Fix (mirrors upstream PR vllm-project/vllm#40860): at the end of
finalize_weights() of each expert module, drop the four original
Parameters by assigning them to None so they are removed from the module's
_parameters dict. transform_weights_for_mega_moe allocates fresh L1 + SF
tensors and only the L2 weight aliases the original w2_weight storage --
_transformed_l2_weights still holds that reference, so the storage stays
live via refcount. PyTorch's caching allocator can then reuse the freed
storage for the NEXT layer's interleave temporaries within the same
finalize loop.

Reference: vllm/model_executor/models/deepseek_v4.py,
DeepseekV4MegaMoEExperts.finalize_weights().
"""

import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py"
)

# Idempotency marker
MARKER = "srt-slurm-sa hotfix: free original MegaMoE expert weights"

# Anchor: closing of the _transformed_l1/l2 assignment in finalize_weights().
# The triple-`)` pattern is unique in the file.
OLD = (
    "        self._transformed_l1_weights, self._transformed_l2_weights = (\n"
    "            deep_gemm.transform_weights_for_mega_moe(\n"
    "                (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),\n"
    "                (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),\n"
    "            )\n"
    "        )\n"
)

NEW = (
    OLD
    + "        # srt-slurm-sa hotfix: free original MegaMoE expert weights.\n"
    + "        # Mirrors vllm-project/vllm#40860. transform_weights_for_mega_moe\n"
    + "        # allocates fresh L1 + SF tensors; only the L2 weight aliases the\n"
    + "        # original w2_weight storage, but _transformed_l2_weights holds that\n"
    + "        # reference, so dropping the Parameters is safe via refcount and the\n"
    + "        # freed storage returns to the caching allocator in time for the next\n"
    + "        # layer's interleave temp (~1 GiB).\n"
    + "        self.w13_weight = None\n"
    + "        self.w13_weight_scale = None\n"
    + "        self.w2_weight = None\n"
    + "        self.w2_weight_scale = None\n"
)


def main():
    if not TARGET.exists():
        print(f"[vllm-mega-moe-free-orig] Target not found: {TARGET}", file=sys.stderr)
        sys.exit(1)

    content = TARGET.read_text()

    if MARKER in content:
        print("[vllm-mega-moe-free-orig] Already patched, skipping.", file=sys.stderr)
        return

    count = content.count(OLD)
    if count == 0:
        print(
            "[vllm-mega-moe-free-orig] Could not find finalize_weights anchor. "
            "vLLM version may have drifted; inspect "
            "DeepseekV4MegaMoEExperts.finalize_weights().",
            file=sys.stderr,
        )
        sys.exit(1)
    if count > 1:
        print(
            f"[vllm-mega-moe-free-orig] Anchor is ambiguous ({count} occurrences); "
            "refusing to patch.",
            file=sys.stderr,
        )
        sys.exit(1)

    content = content.replace(OLD, NEW)
    TARGET.write_text(content)
    print(
        "[vllm-mega-moe-free-orig] Freed original w13/w2 weights and scales "
        "in DeepseekV4MegaMoEExperts.finalize_weights().",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
