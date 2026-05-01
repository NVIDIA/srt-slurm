"""
Patch vLLM to backport Inferact/vllm-svf#180 — bf16 activation support for
the FlashInfer NVLink one-sided MoE all-to-all path.

Without the patch, FlashInferNVLinkOneSidedPrepareAndFinalize hard-codes the
dispatch payload to nvfp4 (0.5 B/elem hidden + per-16-elem fp8 scales). That
crashes for experts that prefer to receive bf16 tokens and quantize
post-dispatch (e.g. trtllm_mxfp4_moe with mxfp8 activations) and for any
non-nvfp4 quant_dtype.

Affected files (from PR diff):
  - vllm/distributed/device_communicators/all2all.py
  - vllm/model_executor/layers/fused_moe/all2all_utils.py
  - vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
  - vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py

Reference: https://github.com/Inferact/vllm-svf/pull/180
"""

import sys
from pathlib import Path

VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")

# --- File 1: distributed/device_communicators/all2all.py ----------------------

ALL2ALL_TARGET = VLLM_ROOT / "distributed/device_communicators/all2all.py"

ALL2ALL_OLD = (
    "        top_k: int,\n"
    "        num_experts: int,\n"
    "        hidden_size: int,\n"
    "    ):\n"
    '        """Initialize the MoeAlltoAll workspace."""\n'
    "        if self.initialized:\n"
    "            return\n"
)

ALL2ALL_NEW = (
    "        top_k: int,\n"
    "        num_experts: int,\n"
    "        hidden_size: int,\n"
    "        dispatch_dtype_bytes_per_elem: int = 0,\n"
    "        dispatch_has_fp8_scale: bool = True,\n"
    "    ):\n"
    '        """Initialize the MoeAlltoAll workspace.\n'
    "\n"
    "        dispatch_dtype_bytes_per_elem: bytes/elem for the dispatched hidden\n"
    "            states. Use 0 as a sentinel for sub-byte nvfp4 (0.5 B/elem); use\n"
    "            1 for fp8, 2 for bf16/fp16.\n"
    "        dispatch_has_fp8_scale: whether a per-16-elem fp8 scale tensor is\n"
    "            dispatched alongside the hidden states (true for nvfp4/fp8,\n"
    "            false for bf16 passthrough).\n"
    '        """\n'
    "        if self.initialized:\n"
    "            return\n"
)

ALL2ALL_OLD_PAYLOAD = (
    "        total_dispatch_payload_size_per_token = (\n"
    "            hidden_size // 2  # nvfp4 hidden states\n"
    "            + hidden_size // 16  # fp8 scaling factors\n"
    "            + top_k * 4  # int32 topks ids\n"
    "            + top_k * 4  # float32 topk weights\n"
    "        )\n"
)

ALL2ALL_NEW_PAYLOAD = (
    "        if dispatch_dtype_bytes_per_elem == 0:\n"
    "            hidden_bytes = hidden_size // 2  # nvfp4\n"
    "        else:\n"
    "            hidden_bytes = hidden_size * dispatch_dtype_bytes_per_elem\n"
    "        scale_bytes = hidden_size // 16 if dispatch_has_fp8_scale else 0\n"
    "        total_dispatch_payload_size_per_token = (\n"
    "            hidden_bytes\n"
    "            + scale_bytes\n"
    "            + top_k * 4  # int32 topks ids\n"
    "            + top_k * 4  # float32 topk weights\n"
    "        )\n"
)

# --- File 2: fused_moe/all2all_utils.py ---------------------------------------

ALL2ALL_UTILS_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/all2all_utils.py"

ALL2ALL_UTILS_OLD_SIG = (
    "    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,\n"
    "    allow_new_interface: bool = False,\n"
    "    use_monolithic: bool = False,\n"
    ") -> FusedMoEPrepareAndFinalize | None:\n"
)

ALL2ALL_UTILS_NEW_SIG = (
    "    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,\n"
    "    allow_new_interface: bool = False,\n"
    "    use_monolithic: bool = False,\n"
    "    defer_input_quant: bool = False,\n"
    ") -> FusedMoEPrepareAndFinalize | None:\n"
)

ALL2ALL_UTILS_OLD_BUILD = (
    "        prepare_finalize = FlashInferNVLinkOneSidedPrepareAndFinalize(\n"
    "            max_num_tokens=max_num_tokens,\n"
    "            top_k=moe.experts_per_token,\n"
    "            num_experts=moe.num_experts,\n"
    "            hidden_size=moe.hidden_dim,\n"
    "            num_dispatchers=all2all_manager.world_size,\n"
    "        )\n"
)

ALL2ALL_UTILS_NEW_BUILD = (
    "        if defer_input_quant or quant_config.quant_dtype is None:\n"
    "            # Experts (e.g. trtllm_mxfp4 with mxfp8 activations) quantize\n"
    "            # post-dispatch; ship bf16 tokens with no per-token scale payload.\n"
    "            dispatch_dtype_bytes_per_elem, dispatch_has_fp8_scale = 2, False\n"
    '        elif quant_config.quant_dtype == "nvfp4":\n'
    "            dispatch_dtype_bytes_per_elem, dispatch_has_fp8_scale = 0, True\n"
    "        else:\n"
    "            raise NotImplementedError(\n"
    '                "flashinfer_nvlink_one_sided dispatch only supports nvfp4, "\n'
    '                "bf16, and defer_input_quant paths today; got "\n'
    '                f"quant_dtype={quant_config.quant_dtype!r}"\n'
    "            )\n"
    "        prepare_finalize = FlashInferNVLinkOneSidedPrepareAndFinalize(\n"
    "            max_num_tokens=max_num_tokens,\n"
    "            top_k=moe.experts_per_token,\n"
    "            num_experts=moe.num_experts,\n"
    "            hidden_size=moe.hidden_dim,\n"
    "            num_dispatchers=all2all_manager.world_size,\n"
    "            dispatch_dtype_bytes_per_elem=dispatch_dtype_bytes_per_elem,\n"
    "            dispatch_has_fp8_scale=dispatch_has_fp8_scale,\n"
    "        )\n"
)

# --- File 3: fused_moe/oracle/mxfp4.py ----------------------------------------

MXFP4_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/oracle/mxfp4.py"

MXFP4_OLD = (
    '    """Create a FusedMoEKernel for the given MXFP4 backend."""\n'
    "    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)\n"
    "\n"
    "    # Create Prepare/Finalize.\n"
    "    prepare_finalize = maybe_make_prepare_finalize(\n"
    "        moe=moe_config,\n"
)

MXFP4_NEW = (
    '    """Create a FusedMoEKernel for the given MXFP4 backend."""\n'
    "    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)\n"
    "\n"
    "    # Some experts (trtllm_mxfp4 with mxfp8 activations) prefer bf16 tokens\n"
    "    # on dispatch and quantize internally; signal this to the prepare/finalize\n"
    "    # so workspace + prepare path ship bf16 instead of the quant_config dtype.\n"
    "    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (\n"
    "        TrtLlmMxfp4ExpertsBase,\n"
    "    )\n"
    "\n"
    "    defer_input_quant = issubclass(experts_cls, TrtLlmMxfp4ExpertsBase)\n"
    "\n"
    "    # Create Prepare/Finalize.\n"
    "    prepare_finalize = maybe_make_prepare_finalize(\n"
    "        moe=moe_config,\n"
)

MXFP4_OLD_CALL = (
    "        routing_tables=routing_tables,\n"
    "        allow_new_interface=True,\n"
    "        use_monolithic=is_monolithic,\n"
    "    )\n"
    "    assert prepare_finalize is not None\n"
)

MXFP4_NEW_CALL = (
    "        routing_tables=routing_tables,\n"
    "        allow_new_interface=True,\n"
    "        use_monolithic=is_monolithic,\n"
    "        defer_input_quant=defer_input_quant,\n"
    "    )\n"
    "    assert prepare_finalize is not None\n"
)

# --- File 4: fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py --------

PREP_TARGET = VLLM_ROOT / (
    "model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py"
)

PREP_OLD_INIT = (
    "        num_experts: int,\n"
    "        hidden_size: int,\n"
    "        num_dispatchers: int = 1,\n"
    "    ):\n"
    "        super().__init__()\n"
)

PREP_NEW_INIT = (
    "        num_experts: int,\n"
    "        hidden_size: int,\n"
    "        num_dispatchers: int = 1,\n"
    "        dispatch_dtype_bytes_per_elem: int = 0,\n"
    "        dispatch_has_fp8_scale: bool = True,\n"
    "    ):\n"
    "        super().__init__()\n"
)

PREP_OLD_CALL = (
    "            top_k=self.top_k,\n"
    "            num_experts=self.num_experts,\n"
    "            hidden_size=self.hidden_size,\n"
    "        )\n"
)

PREP_NEW_CALL = (
    "            top_k=self.top_k,\n"
    "            num_experts=self.num_experts,\n"
    "            hidden_size=self.hidden_size,\n"
    "            dispatch_dtype_bytes_per_elem=dispatch_dtype_bytes_per_elem,\n"
    "            dispatch_has_fp8_scale=dispatch_has_fp8_scale,\n"
    "        )\n"
)

PREP_OLD_QUANT = (
    "        a1q, a1q_scale = moe_kernel_quantize_input(\n"
    "            a1,\n"
    "            quant_config.a1_gscale,\n"
    "            quant_config.quant_dtype,\n"
    "            quant_config.per_act_token_quant,\n"
    "            quant_config.block_shape,\n"
    "            is_fp4_scale_swizzled=False,  # delay swizzle to after comm\n"
    "        )\n"
)

PREP_NEW_QUANT = (
    "        if defer_input_quant:\n"
    "            # Experts (e.g. trtllm_mxfp4_moe with mxfp8 activations) will\n"
    "            # quantize post-dispatch. Ship bf16 tokens and skip scales.\n"
    "            a1q, a1q_scale = a1, None\n"
    "        else:\n"
    "            a1q, a1q_scale = moe_kernel_quantize_input(\n"
    "                a1,\n"
    "                quant_config.a1_gscale,\n"
    "                quant_config.quant_dtype,\n"
    "                quant_config.per_act_token_quant,\n"
    "                quant_config.block_shape,\n"
    "                is_fp4_scale_swizzled=False,  # delay swizzle to after comm\n"
    "            )\n"
)

# (target file, marker indicating already-patched, [(name, old, new), ...])
FILES = [
    (
        ALL2ALL_TARGET,
        "dispatch_dtype_bytes_per_elem",
        [
            ("MoeAlltoAll.initialize signature", ALL2ALL_OLD, ALL2ALL_NEW),
            ("MoeAlltoAll dispatch payload sizing", ALL2ALL_OLD_PAYLOAD, ALL2ALL_NEW_PAYLOAD),
        ],
    ),
    (
        ALL2ALL_UTILS_TARGET,
        # Note: bare "defer_input_quant" appears in a comment in the base
        # file ("# Unquantized dispatch (e.g. AITER with defer_input_quant):"),
        # so we anchor on a string we *introduce* — namely the parameter
        # declaration in maybe_make_prepare_finalize's signature.
        "defer_input_quant: bool = False,",
        [
            ("maybe_make_prepare_finalize signature", ALL2ALL_UTILS_OLD_SIG, ALL2ALL_UTILS_NEW_SIG),
            (
                "FlashInferNVLinkOneSided builder",
                ALL2ALL_UTILS_OLD_BUILD,
                ALL2ALL_UTILS_NEW_BUILD,
            ),
        ],
    ),
    (
        MXFP4_TARGET,
        # Note: bare "TrtLlmMxfp4ExpertsBase" appears in a comment in the
        # base file. Anchor on the assignment we introduce instead.
        "defer_input_quant = issubclass(experts_cls, TrtLlmMxfp4ExpertsBase)",
        [
            ("make_mxfp4_moe_kernel defer_input_quant detection", MXFP4_OLD, MXFP4_NEW),
            ("make_mxfp4_moe_kernel pass-through", MXFP4_OLD_CALL, MXFP4_NEW_CALL),
        ],
    ),
    (
        PREP_TARGET,
        "dispatch_dtype_bytes_per_elem",
        [
            ("FlashInferNVLinkOneSided __init__ signature", PREP_OLD_INIT, PREP_NEW_INIT),
            ("FlashInferNVLinkOneSided initialize call", PREP_OLD_CALL, PREP_NEW_CALL),
            ("FlashInferNVLinkOneSided prepare quant branch", PREP_OLD_QUANT, PREP_NEW_QUANT),
        ],
    ),
]


def patch_file(target: Path, marker: str, patches: list[tuple[str, str, str]]) -> bool:
    if not target.exists():
        print(f"[nvlink-bf16-patch] Target not found: {target}", file=sys.stderr)
        return False

    content = target.read_text()
    if marker in content:
        print(f"[nvlink-bf16-patch] {target.name}: already patched, skipping.", file=sys.stderr)
        return True

    new_content = content
    for name, old, new in patches:
        count = new_content.count(old)
        if count == 0:
            print(
                f"[nvlink-bf16-patch] {target.name}: anchor for {name!r} not found. "
                "vLLM version may have drifted.",
                file=sys.stderr,
            )
            return False
        if count > 1:
            print(
                f"[nvlink-bf16-patch] {target.name}: anchor for {name!r} is ambiguous "
                f"({count} matches); refusing to patch.",
                file=sys.stderr,
            )
            return False
        new_content = new_content.replace(old, new, 1)
        print(f"[nvlink-bf16-patch] {target.name}: patched {name}", file=sys.stderr)

    target.write_text(new_content)
    return True


def main():
    failures = 0
    for target, marker, patches in FILES:
        if not patch_file(target, marker, patches):
            failures += 1

    if failures:
        print(f"[nvlink-bf16-patch] {failures} file(s) failed to patch", file=sys.stderr)
        sys.exit(1)
    print("[nvlink-bf16-patch] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()