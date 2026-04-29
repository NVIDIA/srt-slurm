"""
Patch vLLM v0.20.0 to backport vllm-project/vllm#40960 — bf16 / mxfp8
activation support for the FlashInfer NVLink one-sided MoE all-to-all path.

Without the patch, FlashInferNVLinkOneSidedPrepareAndFinalize hard-codes the
dispatch payload to nvfp4 (0.5 B/elem hidden + per-16-elem fp8 scales) and
the upstream guard rejects any non-nvfp4 quant_dtype outright. The patch
generalizes the dispatch sizing to bytes-per-element + scale-bytes-per-token,
adds an mxfp8 quant branch, removes the nvfp4-only guard, and threads
mx_alignment through the quantizer pipeline so trtllm_mxfp4 with mxfp8
activations can use this backend end-to-end.

Affected files (all from PR 40960):
  - vllm/distributed/device_communicators/all2all.py
  - vllm/model_executor/layers/fused_moe/all2all_utils.py
  - vllm/model_executor/layers/fused_moe/config.py
  - vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py
  - vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
  - vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py
  - vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py
  - vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
  - vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py
  - vllm/model_executor/layers/fused_moe/utils.py
  - vllm/model_executor/layers/quantization/utils/mxfp8_utils.py

Reference: https://github.com/vllm-project/vllm/pull/40960
Target: vLLM v0.20.0

Note: the new flashinfer_nvlink_one_sided.dispatch() call passes
`invalid_token_expert_id` and `expert_id_payload_index` kwargs. The installed
flashinfer must support these (they are used by the same PR's expectations).
"""

import sys
from pathlib import Path

VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")

# =============================================================================
# File 1: distributed/device_communicators/all2all.py
# =============================================================================

ALL2ALL_TARGET = VLLM_ROOT / "distributed/device_communicators/all2all.py"

ALL2ALL_OLD_SIG = (
    "        top_k: int,\n"
    "        num_experts: int,\n"
    "        hidden_size: int,\n"
    "    ):\n"
    '        """Initialize the MoeAlltoAll workspace."""\n'
)

ALL2ALL_NEW_SIG = (
    "        top_k: int,\n"
    "        num_experts: int,\n"
    "        hidden_size: int,\n"
    "        dispatch_dtype_bytes_per_elem: int = 0,\n"
    "        dispatch_scale_bytes_per_token: int = 0,\n"
    "    ):\n"
    '        """Initialize the MoeAlltoAll workspace."""\n'
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
    "            hidden_bytes = hidden_size // 2\n"
    "        else:\n"
    "            hidden_bytes = hidden_size * dispatch_dtype_bytes_per_elem\n"
    "        total_dispatch_payload_size_per_token = (\n"
    "            hidden_bytes\n"
    "            + dispatch_scale_bytes_per_token\n"
    "            + top_k * 4  # int32 topks ids\n"
    "            + top_k * 4  # float32 topk weights\n"
    "        )\n"
)

# =============================================================================
# File 2: model_executor/layers/fused_moe/all2all_utils.py
# =============================================================================

ALL2ALL_UTILS_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/all2all_utils.py"

ALL2ALL_UTILS_OLD = (
    "    elif moe.use_fi_nvl_one_sided_kernels:\n"
    "        assert quant_config is not None\n"
    '        if quant_config.quant_dtype != "nvfp4":\n'
    "            raise ValueError(\n"
    "                \"The 'flashinfer_nvlink_one_sided' all2all backend only \"\n"
    '                "supports nvfp4 activation quantization, but got "\n'
    '                f"quant_dtype={quant_config.quant_dtype!r}. Use a different "\n'
    "                \"all2all backend (e.g. 'flashinfer_nvlink_two_sided' or \"\n"
    "                \"'allgather_reducescatter') for non-nvfp4 models.\"\n"
    "            )\n"
    "        max_num_tokens = (\n"
    "            get_current_vllm_config().scheduler_config.max_num_batched_tokens\n"
    "        )\n"
    "        prepare_finalize = FlashInferNVLinkOneSidedPrepareAndFinalize(\n"
    "            max_num_tokens=max_num_tokens,\n"
    "            top_k=moe.experts_per_token,\n"
    "            num_experts=moe.num_experts,\n"
    "            hidden_size=moe.hidden_dim,\n"
    "            num_dispatchers=all2all_manager.world_size,\n"
    "        )\n"
)

ALL2ALL_UTILS_NEW = (
    "    elif moe.use_fi_nvl_one_sided_kernels:\n"
    "        assert quant_config is not None\n"
    "        max_num_tokens = (\n"
    "            get_current_vllm_config().scheduler_config.max_num_batched_tokens\n"
    "        )\n"
    "        if quant_config.quant_dtype is None:\n"
    "            dispatch_dtype_bytes_per_elem = 2\n"
    "            dispatch_scale_bytes_per_token = 0\n"
    '        elif quant_config.quant_dtype == "nvfp4":\n'
    "            dispatch_dtype_bytes_per_elem = 0\n"
    "            dispatch_scale_bytes_per_token = moe.hidden_dim // 16\n"
    '        elif quant_config.quant_dtype == "mxfp8":\n'
    "            dispatch_dtype_bytes_per_elem = 1\n"
    "            align = quant_config.mx_alignment\n"
    "            if align > 0:\n"
    "                padded_k = ((moe.hidden_dim + align - 1) // align) * align\n"
    "            else:\n"
    "                padded_k = moe.hidden_dim\n"
    "            dispatch_scale_bytes_per_token = padded_k // 32\n"
    "        else:\n"
    "            raise NotImplementedError(\n"
    '                "flashinfer_nvlink_one_sided dispatch supports nvfp4, mxfp8, "\n'
    '                "and bf16 (quant_dtype=None) today; got "\n'
    '                f"quant_dtype={quant_config.quant_dtype!r}"\n'
    "            )\n"
    "        prepare_finalize = FlashInferNVLinkOneSidedPrepareAndFinalize(\n"
    "            max_num_tokens=max_num_tokens,\n"
    "            top_k=moe.experts_per_token,\n"
    "            num_experts=moe.num_experts,\n"
    "            hidden_size=moe.hidden_dim,\n"
    "            num_dispatchers=all2all_manager.world_size,\n"
    "            dispatch_dtype_bytes_per_elem=dispatch_dtype_bytes_per_elem,\n"
    "            dispatch_scale_bytes_per_token=dispatch_scale_bytes_per_token,\n"
    "        )\n"
)

# =============================================================================
# File 3: model_executor/layers/fused_moe/config.py
# =============================================================================

CONFIG_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/config.py"

CONFIG_OLD_FIELD = (
    "    gemm1_clamp_limit: float | None = None\n"
    "\n"
    "    def __post_init__(self):\n"
)

CONFIG_NEW_FIELD = (
    "    gemm1_clamp_limit: float | None = None\n"
    "\n"
    "    mx_alignment: int = 0\n"
    "\n"
    "    def __post_init__(self):\n"
)

CONFIG_OLD_HELPER_SIG = (
    "def mxfp4_mxfp8_moe_quant_config(\n"
    '    w1_scale: Union[torch.Tensor, "PrecisionConfig"],\n'
    '    w2_scale: Union[torch.Tensor, "PrecisionConfig"],\n'
    "    a1_scale: torch.Tensor | None = None,\n"
    "    a2_scale: torch.Tensor | None = None,\n"
    "    w1_bias: torch.Tensor | None = None,\n"
    "    w2_bias: torch.Tensor | None = None,\n"
    "    block_shape: list[int] | None = None,\n"
    "    gemm1_alpha: float | None = None,\n"
    "    gemm1_beta: float | None = None,\n"
    "    gemm1_clamp_limit: float | None = None,\n"
    ") -> FusedMoEQuantConfig:\n"
)

CONFIG_NEW_HELPER_SIG = (
    "def mxfp4_mxfp8_moe_quant_config(\n"
    '    w1_scale: Union[torch.Tensor, "PrecisionConfig"],\n'
    '    w2_scale: Union[torch.Tensor, "PrecisionConfig"],\n'
    "    a1_scale: torch.Tensor | None = None,\n"
    "    a2_scale: torch.Tensor | None = None,\n"
    "    w1_bias: torch.Tensor | None = None,\n"
    "    w2_bias: torch.Tensor | None = None,\n"
    "    block_shape: list[int] | None = None,\n"
    "    gemm1_alpha: float | None = None,\n"
    "    gemm1_beta: float | None = None,\n"
    "    gemm1_clamp_limit: float | None = None,\n"
    "    mx_alignment: int = 0,\n"
    ") -> FusedMoEQuantConfig:\n"
)

CONFIG_OLD_HELPER_BODY = (
    "        gemm1_alpha=gemm1_alpha,\n"
    "        gemm1_beta=gemm1_beta,\n"
    "        gemm1_clamp_limit=gemm1_clamp_limit,\n"
    "    )\n"
    "\n"
    "\n"
    "def mxfp4_w4a8_moe_quant_config(\n"
)

CONFIG_NEW_HELPER_BODY = (
    "        gemm1_alpha=gemm1_alpha,\n"
    "        gemm1_beta=gemm1_beta,\n"
    "        gemm1_clamp_limit=gemm1_clamp_limit,\n"
    "        mx_alignment=mx_alignment,\n"
    "    )\n"
    "\n"
    "\n"
    "def mxfp4_w4a8_moe_quant_config(\n"
)

# =============================================================================
# File 4: model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py
# =============================================================================

TRTLLM_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py"

# 4a) Drop the use_mxfp8_input cached attribute on the base class.
TRTLLM_OLD_USE_FLAG = (
    "        self.max_capture_size = (\n"
    "            get_current_vllm_config().compilation_config.max_cudagraph_capture_size\n"
    "        )\n"
    "\n"
    "        # P1-5 fix: use public quant_dtype property instead of private _a1\n"
    '        self.use_mxfp8_input = quant_config.quant_dtype == "mxfp8"\n'
    "\n"
    "    @staticmethod\n"
    "    def _supports_current_device() -> bool:\n"
)

TRTLLM_NEW_USE_FLAG = (
    "        self.max_capture_size = (\n"
    "            get_current_vllm_config().compilation_config.max_cudagraph_capture_size\n"
    "        )\n"
    "\n"
    "    @staticmethod\n"
    "    def _supports_current_device() -> bool:\n"
)

# 4b) Base-class expects_unquantized_inputs flips True -> False, comment dropped.
TRTLLM_OLD_BASE_EUI = (
    "    @property\n"
    "    def expects_unquantized_inputs(self) -> bool:\n"
    "        # Expert handles MXFP8 quantization internally if needed\n"
    "        return True\n"
    "\n"
    "\n"
    "class TrtLlmMxfp4ExpertsMonolithic(\n"
)

TRTLLM_NEW_BASE_EUI = (
    "    @property\n"
    "    def expects_unquantized_inputs(self) -> bool:\n"
    "        return False\n"
    "\n"
    "\n"
    "class TrtLlmMxfp4ExpertsMonolithic(\n"
)

# 4c) Monolithic.apply quant block.
TRTLLM_OLD_APPLY_MONO = (
    "        from flashinfer import trtllm_fp4_block_scale_moe\n"
    "\n"
    "        # Handle input quantization\n"
    "        if self.use_mxfp8_input:\n"
    "            from flashinfer import mxfp8_quantize\n"
    "\n"
    "            x_quant, x_scale = mxfp8_quantize(\n"
    "                hidden_states,\n"
    "                is_sf_swizzled_layout=False,\n"
    "                alignment=256,\n"
    "            )\n"
    "            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(\n"
    "                *hidden_states.shape[:-1], -1\n"
    "            )\n"
    "        else:\n"
    "            assert hidden_states.dtype == torch.bfloat16\n"
    "            x_quant = hidden_states\n"
    "            x_scale = None\n"
    "\n"
    "        output = torch.empty_like(hidden_states)\n"
)

TRTLLM_NEW_APPLY_MONO = (
    "        from flashinfer import trtllm_fp4_block_scale_moe\n"
    "\n"
    "        if a1q_scale is not None:\n"
    "            x_quant = hidden_states\n"
    "            x_scale = a1q_scale.view(torch.float8_e4m3fn).reshape(\n"
    "                *hidden_states.shape[:-1], -1\n"
    "            )\n"
    "        else:\n"
    "            assert hidden_states.dtype == torch.bfloat16\n"
    "            x_quant = hidden_states\n"
    "            x_scale = None\n"
    "        output = torch.empty(\n"
    "            *hidden_states.shape[:-1],\n"
    "            self.hidden_dim,\n"
    "            dtype=torch.bfloat16,\n"
    "            device=hidden_states.device,\n"
    "        )\n"
)

# 4d) Drop Modular's expects_unquantized_inputs override entirely.
TRTLLM_OLD_MOD_EUI = (
    '    Moved from trtllm_moe.py.\n'
    '    """\n'
    "\n"
    "    @property\n"
    "    def expects_unquantized_inputs(self) -> bool:\n"
    "        return True\n"
    "\n"
    "    @staticmethod\n"
    "    def _supports_parallel_config(\n"
)

TRTLLM_NEW_MOD_EUI = (
    '    Moved from trtllm_moe.py.\n'
    '    """\n'
    "\n"
    "    @staticmethod\n"
    "    def _supports_parallel_config(\n"
)

# 4e) Modular.apply quant block (preceded by local_expert_offset, followed by
#     the topk pack comment — anchors give uniqueness vs the monolithic block).
TRTLLM_OLD_APPLY_MOD = (
    "        local_expert_offset = self.moe_config.ep_rank * local_num_experts\n"
    "\n"
    "        # Handle input quantization\n"
    "        if self.use_mxfp8_input:\n"
    "            from flashinfer import mxfp8_quantize\n"
    "\n"
    "            x_quant, x_scale = mxfp8_quantize(\n"
    "                hidden_states,\n"
    "                is_sf_swizzled_layout=False,\n"
    "                alignment=256,\n"
    "            )\n"
    "            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(\n"
    "                *hidden_states.shape[:-1], -1\n"
    "            )\n"
    "        else:\n"
    "            assert hidden_states.dtype == torch.bfloat16\n"
    "            x_quant = hidden_states\n"
    "            x_scale = None\n"
    "\n"
    "        # Pack topk ids and weights into format expected by the kernel.\n"
)

TRTLLM_NEW_APPLY_MOD = (
    "        local_expert_offset = self.moe_config.ep_rank * local_num_experts\n"
    "\n"
    "        if a1q_scale is not None:\n"
    "            x_quant = hidden_states\n"
    "            x_scale = a1q_scale.view(torch.float8_e4m3fn).reshape(\n"
    "                *hidden_states.shape[:-1], -1\n"
    "            )\n"
    "        else:\n"
    "            assert hidden_states.dtype == torch.bfloat16\n"
    "            x_quant = hidden_states\n"
    "            x_scale = None\n"
    "\n"
    "        # Pack topk ids and weights into format expected by the kernel.\n"
)

# =============================================================================
# File 5: model_executor/layers/fused_moe/oracle/mxfp4.py
# =============================================================================

MXFP4_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/oracle/mxfp4.py"

# 5a) Split the merged TRTLLM/CUTLASS branch so TRTLLM gets mx_alignment=256.
MXFP4_OLD_BRANCH = (
    "    elif mxfp4_backend in (\n"
    "        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,\n"
    "        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,\n"
    "    ):\n"
    "        return mxfp4_mxfp8_moe_quant_config(\n"
    "            w1_bias=w1_bias,\n"
    "            w2_bias=w2_bias,\n"
    "            w1_scale=w1_scale,\n"
    "            w2_scale=w2_scale,\n"
    "            gemm1_alpha=gemm1_alpha,\n"
    "            gemm1_beta=gemm1_beta,\n"
    "            gemm1_clamp_limit=swiglu_limit,\n"
    "        )\n"
)

MXFP4_NEW_BRANCH = (
    "    elif mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8:\n"
    "        return mxfp4_mxfp8_moe_quant_config(\n"
    "            w1_bias=w1_bias,\n"
    "            w2_bias=w2_bias,\n"
    "            w1_scale=w1_scale,\n"
    "            w2_scale=w2_scale,\n"
    "            gemm1_alpha=gemm1_alpha,\n"
    "            gemm1_beta=gemm1_beta,\n"
    "            gemm1_clamp_limit=swiglu_limit,\n"
    "            mx_alignment=256,\n"
    "        )\n"
    "    elif mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8:\n"
    "        return mxfp4_mxfp8_moe_quant_config(\n"
    "            w1_bias=w1_bias,\n"
    "            w2_bias=w2_bias,\n"
    "            w1_scale=w1_scale,\n"
    "            w2_scale=w2_scale,\n"
    "            gemm1_alpha=gemm1_alpha,\n"
    "            gemm1_beta=gemm1_beta,\n"
    "            gemm1_clamp_limit=swiglu_limit,\n"
    "        )\n"
)

# 5b) Drop the now-redundant comment in make_mxfp4_moe_kernel.
MXFP4_OLD_COMMENT = (
    "    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)\n"
    "\n"
    "    # Create Prepare/Finalize.\n"
    "    prepare_finalize = maybe_make_prepare_finalize(\n"
)

MXFP4_NEW_COMMENT = (
    "    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)\n"
    "\n"
    "    prepare_finalize = maybe_make_prepare_finalize(\n"
)

# =============================================================================
# File 6: model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py
# =============================================================================

PREP_TARGET = VLLM_ROOT / (
    "model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py"
)

PREP_OLD_INIT_SIG = (
    "        num_experts: int,\n"
    "        hidden_size: int,\n"
    "        num_dispatchers: int = 1,\n"
    "    ):\n"
    "        super().__init__()\n"
    "        self.max_num_tokens = max_num_tokens\n"
)

PREP_NEW_INIT_SIG = (
    "        num_experts: int,\n"
    "        hidden_size: int,\n"
    "        num_dispatchers: int = 1,\n"
    "        dispatch_dtype_bytes_per_elem: int = 0,\n"
    "        dispatch_scale_bytes_per_token: int = 0,\n"
    "    ):\n"
    "        super().__init__()\n"
    "        self.max_num_tokens = max_num_tokens\n"
)

PREP_OLD_INIT_BODY = (
    "        self.num_experts = num_experts\n"
    "        self.hidden_size = hidden_size\n"
    "        self.num_dispatchers_ = num_dispatchers\n"
    "\n"
    "        device_communicator = get_ep_group().device_communicator\n"
)

PREP_NEW_INIT_BODY = (
    "        self.num_experts = num_experts\n"
    "        self.hidden_size = hidden_size\n"
    "        self.num_dispatchers_ = num_dispatchers\n"
    "        self.scale_elems_per_token = dispatch_scale_bytes_per_token\n"
    "\n"
    "        device_communicator = get_ep_group().device_communicator\n"
)

PREP_OLD_INITIALIZE_CALL = (
    "            top_k=self.top_k,\n"
    "            num_experts=self.num_experts,\n"
    "            hidden_size=self.hidden_size,\n"
    "        )\n"
)

PREP_NEW_INITIALIZE_CALL = (
    "            top_k=self.top_k,\n"
    "            num_experts=self.num_experts,\n"
    "            hidden_size=self.hidden_size,\n"
    "            dispatch_dtype_bytes_per_elem=dispatch_dtype_bytes_per_elem,\n"
    "            dispatch_scale_bytes_per_token=dispatch_scale_bytes_per_token,\n"
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
    "\n"
    "        payloads = []\n"
    "        payloads.append(a1q)\n"
    "        if a1q_scale is not None:\n"
    "            payloads.append(a1q_scale)\n"
    "        payloads.append(topk_ids)\n"
    "        payloads.append(topk_weights)\n"
)

PREP_NEW_QUANT = (
    "        if defer_input_quant:\n"
    "            a1q, a1q_scale = a1, None\n"
    "        else:\n"
    "            a1q, a1q_scale = moe_kernel_quantize_input(\n"
    "                a1,\n"
    "                quant_config.a1_gscale,\n"
    "                quant_config.quant_dtype,\n"
    "                quant_config.per_act_token_quant,\n"
    "                quant_config.block_shape,\n"
    "                is_fp4_scale_swizzled=False,  # delay swizzle to after comm\n"
    "                mx_alignment=quant_config.mx_alignment,\n"
    "            )\n"
    "\n"
    "        payloads = []\n"
    "        payloads.append(a1q)\n"
    "        if a1q_scale is not None:\n"
    "            payloads.append(a1q_scale)\n"
    "        topk_ids_payload_index = len(payloads)\n"
    "        payloads.append(topk_ids)\n"
    "        payloads.append(topk_weights)\n"
)

PREP_OLD_DISPATCH = (
    "            token_selected_experts=topk_ids,\n"
    "            input_payloads=payloads,\n"
    "            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,\n"
    "        )\n"
)

PREP_NEW_DISPATCH = (
    "            token_selected_experts=topk_ids,\n"
    "            input_payloads=payloads,\n"
    "            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,\n"
    "            invalid_token_expert_id=num_experts,\n"
    "            expert_id_payload_index=topk_ids_payload_index,\n"
    "        )\n"
)

PREP_OLD_SCALE_VIEW = (
    "                a1q_scale_recv = nvfp4_block_scale_interleave(a1q_scale_recv)\n"
    "            a1q_scale_recv = a1q_scale_recv.view(-1, self.hidden_size // 16)\n"
)

PREP_NEW_SCALE_VIEW = (
    "                a1q_scale_recv = nvfp4_block_scale_interleave(a1q_scale_recv)\n"
    "            assert self.scale_elems_per_token > 0\n"
    "            a1q_scale_recv = a1q_scale_recv.view(-1, self.scale_elems_per_token)\n"
)

# =============================================================================
# File 7: model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py
# =============================================================================

PREP_TWO_TARGET = VLLM_ROOT / (
    "model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py"
)

PREP_TWO_OLD = (
    "            # NOTE: swizzling pads the scales to multiple of 128\n"
    "            # which makes the scales tensor different shape than\n"
    "            # the hidden states, breaking the A2A kernel. So, we\n"
    "            # delay the swizzling until after the A2A.\n"
    "            is_fp4_scale_swizzled=False,\n"
    "        )\n"
)

PREP_TWO_NEW = (
    "            # NOTE: swizzling pads the scales to multiple of 128\n"
    "            # which makes the scales tensor different shape than\n"
    "            # the hidden states, breaking the A2A kernel. So, we\n"
    "            # delay the swizzling until after the A2A.\n"
    "            is_fp4_scale_swizzled=False,\n"
    "            mx_alignment=quant_config.mx_alignment,\n"
    "        )\n"
)

# =============================================================================
# File 8: model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
# =============================================================================

NAIVE_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py"

NAIVE_OLD = (
    "            per_act_token_quant=quant_config.per_act_token_quant,\n"
    "            block_shape=quant_config.block_shape,\n"
    "            is_fp4_scale_swizzled=False,\n"
    "        )\n"
)

NAIVE_NEW = (
    "            per_act_token_quant=quant_config.per_act_token_quant,\n"
    "            block_shape=quant_config.block_shape,\n"
    "            is_fp4_scale_swizzled=False,\n"
    "            mx_alignment=quant_config.mx_alignment,\n"
    "        )\n"
)

# =============================================================================
# File 9: model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py
# =============================================================================

NODP_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py"

NODP_OLD = (
    "        per_act_token_quant=quant_config.per_act_token_quant,\n"
    "        block_shape=quant_config.block_shape,\n"
    "        is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,\n"
    "    )\n"
)

NODP_NEW = (
    "        per_act_token_quant=quant_config.per_act_token_quant,\n"
    "        block_shape=quant_config.block_shape,\n"
    "        is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,\n"
    "        mx_alignment=quant_config.mx_alignment,\n"
    "    )\n"
)

# =============================================================================
# File 10: model_executor/layers/fused_moe/utils.py
# =============================================================================

UTILS_TARGET = VLLM_ROOT / "model_executor/layers/fused_moe/utils.py"

UTILS_OLD_MXFP8_FN = (
    "def _mxfp8_e4m3_quantize(\n"
    "    A: torch.Tensor,\n"
    "    A_scale: torch.Tensor | None,\n"
    "    per_act_token_quant: bool,\n"
    "    block_shape: list[int] | None = None,\n"
    "    is_sf_swizzled_layout: bool = False,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    "    assert A_scale is None\n"
    "    assert not per_act_token_quant\n"
    "    assert block_shape is None or block_shape == [1, 32]\n"
    "    return mxfp8_e4m3_quantize(A, is_sf_swizzled_layout)\n"
)

UTILS_NEW_MXFP8_FN = (
    "def _mxfp8_e4m3_quantize(\n"
    "    A: torch.Tensor,\n"
    "    A_scale: torch.Tensor | None,\n"
    "    per_act_token_quant: bool,\n"
    "    block_shape: list[int] | None = None,\n"
    "    is_sf_swizzled_layout: bool = False,\n"
    "    mx_alignment: int = 0,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    "    assert A_scale is None\n"
    "    assert not per_act_token_quant\n"
    "    assert block_shape is None or block_shape == [1, 32]\n"
    "    return mxfp8_e4m3_quantize(A, is_sf_swizzled_layout, mx_alignment)\n"
)

UTILS_OLD_KERNEL_SIG = (
    "    is_fp4_scale_swizzled: bool = True,\n"
    "    ocp_mx_scheme: str | None = None,\n"
    "    quantization_emulation: bool = False,\n"
    ") -> tuple[torch.Tensor, torch.Tensor | None]:\n"
)

UTILS_NEW_KERNEL_SIG = (
    "    is_fp4_scale_swizzled: bool = True,\n"
    "    ocp_mx_scheme: str | None = None,\n"
    "    quantization_emulation: bool = False,\n"
    "    mx_alignment: int = 0,\n"
    ") -> tuple[torch.Tensor, torch.Tensor | None]:\n"
)

UTILS_OLD_KERNEL_BODY = (
    "        return _mxfp8_e4m3_quantize(\n"
    "            A,\n"
    "            A_scale,\n"
    "            per_act_token_quant,\n"
    "            block_shape,\n"
    "            is_sf_swizzled_layout=is_fp4_scale_swizzled,\n"
    "        )\n"
)

UTILS_NEW_KERNEL_BODY = (
    "        return _mxfp8_e4m3_quantize(\n"
    "            A,\n"
    "            A_scale,\n"
    "            per_act_token_quant,\n"
    "            block_shape,\n"
    "            is_sf_swizzled_layout=is_fp4_scale_swizzled,\n"
    "            mx_alignment=mx_alignment,\n"
    "        )\n"
)

# =============================================================================
# File 11: model_executor/layers/quantization/utils/mxfp8_utils.py
# =============================================================================

MXFP8_TARGET = VLLM_ROOT / "model_executor/layers/quantization/utils/mxfp8_utils.py"

MXFP8_OLD_IMPL = (
    "def _mxfp8_e4m3_quantize_impl(\n"
    "    x: torch.Tensor, is_sf_swizzled_layout: bool = False\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    "    from vllm.platforms import current_platform\n"
    "\n"
    "    if current_platform.has_device_capability(100):\n"
    "        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize\n"
    "\n"
    "        x_q, x_scales = flashinfer_mxfp8_quantize(\n"
    "            x, is_sf_swizzled_layout=is_sf_swizzled_layout\n"
    "        )\n"
)

MXFP8_NEW_IMPL = (
    "def _mxfp8_e4m3_quantize_impl(\n"
    "    x: torch.Tensor,\n"
    "    is_sf_swizzled_layout: bool = False,\n"
    "    alignment: int = 0,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    "    from vllm.platforms import current_platform\n"
    "\n"
    "    if current_platform.has_device_capability(100):\n"
    "        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize\n"
    "\n"
    "        x_q, x_scales = flashinfer_mxfp8_quantize(\n"
    "            x,\n"
    "            is_sf_swizzled_layout=is_sf_swizzled_layout,\n"
    "            alignment=alignment if alignment > 0 else None,\n"
    "        )\n"
)

MXFP8_OLD_PUBLIC = (
    "def mxfp8_e4m3_quantize(\n"
    "    x: torch.Tensor, is_sf_swizzled_layout: bool = False\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    "    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout)\n"
)

MXFP8_NEW_PUBLIC = (
    "def mxfp8_e4m3_quantize(\n"
    "    x: torch.Tensor,\n"
    "    is_sf_swizzled_layout: bool = False,\n"
    "    alignment: int = 0,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    "    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout, alignment)\n"
)

MXFP8_OLD_FAKE = (
    "def mxfp8_e4m3_quantize_fake(\n"
    "    x: torch.Tensor, is_sf_swizzled_layout: bool = False\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    '    """Fake implementation for torch.compile tracing."""\n'
)

MXFP8_NEW_FAKE = (
    "def mxfp8_e4m3_quantize_fake(\n"
    "    x: torch.Tensor,\n"
    "    is_sf_swizzled_layout: bool = False,\n"
    "    alignment: int = 0,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
    '    """Fake implementation for torch.compile tracing."""\n'
)

# =============================================================================
# File table
# =============================================================================

# (target file, marker indicating already-patched, [(name, old, new), ...])
FILES = [
    (
        ALL2ALL_TARGET,
        "dispatch_scale_bytes_per_token",
        [
            ("MoeAlltoAll.initialize signature", ALL2ALL_OLD_SIG, ALL2ALL_NEW_SIG),
            ("MoeAlltoAll dispatch payload sizing", ALL2ALL_OLD_PAYLOAD, ALL2ALL_NEW_PAYLOAD),
        ],
    ),
    (
        ALL2ALL_UTILS_TARGET,
        "dispatch_scale_bytes_per_token=dispatch_scale_bytes_per_token",
        [
            (
                "drop nvfp4 guard + add nvfp4/mxfp8/None branches",
                ALL2ALL_UTILS_OLD,
                ALL2ALL_UTILS_NEW,
            ),
        ],
    ),
    (
        CONFIG_TARGET,
        "    mx_alignment: int = 0\n",
        [
            ("FusedMoEQuantConfig mx_alignment field", CONFIG_OLD_FIELD, CONFIG_NEW_FIELD),
            ("mxfp4_mxfp8_moe_quant_config signature", CONFIG_OLD_HELPER_SIG, CONFIG_NEW_HELPER_SIG),
            ("mxfp4_mxfp8_moe_quant_config body", CONFIG_OLD_HELPER_BODY, CONFIG_NEW_HELPER_BODY),
        ],
    ),
    (
        TRTLLM_TARGET,
        "if a1q_scale is not None:\n            x_quant = hidden_states",
        [
            ("drop self.use_mxfp8_input flag", TRTLLM_OLD_USE_FLAG, TRTLLM_NEW_USE_FLAG),
            ("base expects_unquantized_inputs -> False", TRTLLM_OLD_BASE_EUI, TRTLLM_NEW_BASE_EUI),
            ("Monolithic.apply quant block", TRTLLM_OLD_APPLY_MONO, TRTLLM_NEW_APPLY_MONO),
            ("drop Modular.expects_unquantized_inputs", TRTLLM_OLD_MOD_EUI, TRTLLM_NEW_MOD_EUI),
            ("Modular.apply quant block", TRTLLM_OLD_APPLY_MOD, TRTLLM_NEW_APPLY_MOD),
        ],
    ),
    (
        MXFP4_TARGET,
        "mx_alignment=256,",
        [
            ("split TRTLLM/CUTLASS branches", MXFP4_OLD_BRANCH, MXFP4_NEW_BRANCH),
            ("drop redundant comment in make_mxfp4_moe_kernel", MXFP4_OLD_COMMENT, MXFP4_NEW_COMMENT),
        ],
    ),
    (
        PREP_TARGET,
        "self.scale_elems_per_token",
        [
            ("FlashInferNVLinkOneSided __init__ signature", PREP_OLD_INIT_SIG, PREP_NEW_INIT_SIG),
            ("FlashInferNVLinkOneSided __init__ body", PREP_OLD_INIT_BODY, PREP_NEW_INIT_BODY),
            ("FlashInferNVLinkOneSided initialize call", PREP_OLD_INITIALIZE_CALL, PREP_NEW_INITIALIZE_CALL),
            ("FlashInferNVLinkOneSided prepare quant + payloads", PREP_OLD_QUANT, PREP_NEW_QUANT),
            ("FlashInferNVLinkOneSided dispatch call", PREP_OLD_DISPATCH, PREP_NEW_DISPATCH),
            ("FlashInferNVLinkOneSided scale view", PREP_OLD_SCALE_VIEW, PREP_NEW_SCALE_VIEW),
        ],
    ),
    (
        PREP_TWO_TARGET,
        "mx_alignment=quant_config.mx_alignment,",
        [
            ("flashinfer_nvlink_two_sided pass mx_alignment", PREP_TWO_OLD, PREP_TWO_NEW),
        ],
    ),
    (
        NAIVE_TARGET,
        "mx_alignment=quant_config.mx_alignment,",
        [
            ("naive_dp_ep pass mx_alignment", NAIVE_OLD, NAIVE_NEW),
        ],
    ),
    (
        NODP_TARGET,
        "mx_alignment=quant_config.mx_alignment,",
        [
            ("no_dp_ep pass mx_alignment", NODP_OLD, NODP_NEW),
        ],
    ),
    (
        UTILS_TARGET,
        "    mx_alignment: int = 0,\n",
        [
            ("_mxfp8_e4m3_quantize add mx_alignment", UTILS_OLD_MXFP8_FN, UTILS_NEW_MXFP8_FN),
            ("moe_kernel_quantize_input signature", UTILS_OLD_KERNEL_SIG, UTILS_NEW_KERNEL_SIG),
            ("moe_kernel_quantize_input mxfp8 call", UTILS_OLD_KERNEL_BODY, UTILS_NEW_KERNEL_BODY),
        ],
    ),
    (
        MXFP8_TARGET,
        "    alignment: int = 0,\n",
        [
            ("_mxfp8_e4m3_quantize_impl alignment", MXFP8_OLD_IMPL, MXFP8_NEW_IMPL),
            ("mxfp8_e4m3_quantize alignment", MXFP8_OLD_PUBLIC, MXFP8_NEW_PUBLIC),
            ("mxfp8_e4m3_quantize_fake alignment", MXFP8_OLD_FAKE, MXFP8_NEW_FAKE),
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
