#!/usr/bin/env python3
"""Revert vLLM PR #41015's FP32->FP4 cvt path inside an installed wheel.

This restores the pre-#41015 Triton nibble quantization for the DeepSeek V4
MXFP4 indexer/compressor kernels. It is intended as a one-off perf bisection
patch for the CUDA 13 nightly image.
"""

from __future__ import annotations

import sys
from pathlib import Path


INDEXER_REL = "vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py"
COMPRESS_REL = "vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py"


OLD_FP4_HELPER = '''@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
'''

NEW_FP4_HELPER = '''@triton.jit
def _e2m1_nibble(x):
    """Quantize fp32 x (already scale-divided) to E2M1 4-bit nibble in uint8.
    Matches torch.bucketize with boundaries
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] and right=False (each boundary
    belongs to the lower bucket), plus sign bit."""
    abs_x = tl.minimum(tl.abs(x), 6.0)
    code = tl.where(
        abs_x <= 0.25,
        0.0,
        tl.where(
            abs_x <= 0.75,
            1.0,
            tl.where(
                abs_x <= 1.25,
                2.0,
                tl.where(
                    abs_x <= 1.75,
                    3.0,
                    tl.where(
                        abs_x <= 2.5,
                        4.0,
                        tl.where(abs_x <= 3.5, 5.0, tl.where(abs_x <= 5.0, 6.0, 7.0)),
                    ),
                ),
            ),
        ),
    )
    code_u8 = code.to(tl.uint8)
    sign = ((x < 0) & (code_u8 != 0)).to(tl.uint8)
    return code_u8 | (sign << 3)
'''

OLD_INDEXER_SCALE = '''    # 6 * 2^-126 is from https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py#L163
    amax = tl.maximum(amax, 6.0 * (2**-126))
    # ue8m0 block scale: 2^ceil(log2(amax/6.0)).
    log2_ratio = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
'''

NEW_INDEXER_SCALE = '''    amax = tl.maximum(amax, 1e-4)
    # ue8m0 block scale: 2^ceil(log2(amax/6.0)).
    log2_ratio = tl.math.ceil(tl.math.log2(amax / 6.0))
'''

OLD_INDEXER_PACK = '''    inv_scale = 1.0 / scale
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale)
    return packed, ue8m0
'''

NEW_INDEXER_PACK = '''    inv_scale = 1.0 / scale
    lo_nib = _e2m1_nibble(x_lo * inv_scale)
    hi_nib = _e2m1_nibble(x_hi * inv_scale)
    packed = lo_nib | (hi_nib << 4)
    return packed, ue8m0
'''

OLD_COMPRESS_QUANT = '''    amax = tl.maximum(amax, 6.0 * (2**-126))

    # ue8m0 block scale: 2^ceil(log2(amax / 6.0)), stored as (exp + 127) byte.
    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [N_QUANT_BLOCKS]

    inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
    packed = _fp32x2_to_fp4x2(
        even_2d * inv_scale_col, odd_2d * inv_scale_col
    )  # (N_BLOCKS, HALF_BLOCK) uint8
    packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))
'''

NEW_COMPRESS_QUANT = '''    amax = tl.maximum(amax, 1e-4)

    # ue8m0 block scale: 2^ceil(log2(amax / 6.0)), stored as (exp + 127) byte.
    log2_ratio = tl.ceil(tl.log2(amax / 6.0))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [N_QUANT_BLOCKS]

    inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
    lo_nib = _e2m1_nibble(even_2d * inv_scale_col)  # (N_BLOCKS, HALF_BLOCK) uint8
    hi_nib = _e2m1_nibble(odd_2d * inv_scale_col)
    packed = lo_nib | (hi_nib << 4)
    packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))
'''


def find_installed_file(relpath: str) -> Path:
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / relpath
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find installed {relpath} on sys.path")


def replace_or_verify(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old in text:
        return text.replace(old, new, 1), True
    if new in text:
        print(f"[vllm_revert_pr41015] {label}: already reverted")
        return text, False
    raise RuntimeError(f"Could not find expected #41015 hunk for {label}")


def patch_indexer(path: Path) -> bool:
    text = path.read_text()
    changed = False
    for old, new, label in (
        (OLD_FP4_HELPER, NEW_FP4_HELPER, "indexer helper"),
        (OLD_INDEXER_SCALE, NEW_INDEXER_SCALE, "indexer scale"),
        (OLD_INDEXER_PACK, NEW_INDEXER_PACK, "indexer pack"),
    ):
        text, did_change = replace_or_verify(text, old, new, label)
        changed = changed or did_change
    if changed:
        path.write_text(text)
        print(f"[vllm_revert_pr41015] patched {path}")
    return changed


def patch_compressor(path: Path) -> bool:
    text = path.read_text()
    changed = False
    text, did_change = replace_or_verify(
        text,
        "from .fused_indexer_q import _fp32x2_to_fp4x2\n",
        "from .fused_indexer_q import _e2m1_nibble\n",
        "compressor import",
    )
    changed = changed or did_change
    text, did_change = replace_or_verify(text, OLD_COMPRESS_QUANT, NEW_COMPRESS_QUANT, "compressor quant")
    changed = changed or did_change
    if changed:
        path.write_text(text)
        print(f"[vllm_revert_pr41015] patched {path}")
    return changed


def main() -> None:
    indexer = find_installed_file(INDEXER_REL)
    compressor = find_installed_file(COMPRESS_REL)
    indexer_changed = patch_indexer(indexer)
    compressor_changed = patch_compressor(compressor)
    changed = indexer_changed or compressor_changed
    if not changed:
        print("[vllm_revert_pr41015] no changes needed")


if __name__ == "__main__":
    main()
