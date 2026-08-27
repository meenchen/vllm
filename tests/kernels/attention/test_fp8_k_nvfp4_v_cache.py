# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache
from vllm import envs
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import fp8_k_nvfp4_v_cache_split_views
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
    trtllm_prefill_attn_fp8_k_nvfp4_v_unpack,
    use_staged_fp8_k_nvfp4_v_prefill,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FP8-K/NVFP4-V TRTLLM-gen attention requires SM100",
)


def _to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = float(x.abs().amax().item() / torch.finfo(torch.float8_e4m3fn).max)
    return (x / scale).to(torch.float8_e4m3fn), scale


def test_fp8_k_nvfp4_v_query_dtype_is_e4m3() -> None:
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(disable_flashinfer_q_quantization=False)
    )
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    builder.cache_dtype = "fp8_k_nvfp4_v"

    assert builder.get_q_data_type(is_prefill=True) == torch.float8_e4m3fn
    assert builder.get_q_data_type(is_prefill=False) == torch.float8_e4m3fn


def test_fp8_k_nvfp4_v_rejects_sm107() -> None:
    kwargs = dict(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8_k_nvfp4_v",
        block_size=64,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
    )
    assert (
        FlashInferBackend.supports_combination(
            **kwargs, device_capability=DeviceCapability(10, 3)
        )
        is None
    )
    reason = FlashInferBackend.supports_combination(
        **kwargs, device_capability=DeviceCapability(10, 7)
    )
    assert reason == "fp8_k_nvfp4_v is not supported on SM107"


@pytest.mark.parametrize("head_size", [128, 256])
def test_fp8_k_nvfp4_v_accepts_supported_head_sizes(head_size: int) -> None:
    assert (
        FlashInferBackend.supports_combination(
            head_size=head_size,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_k_nvfp4_v",
            block_size=64,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            device_capability=DeviceCapability(10, 3),
        )
        is None
    )


def test_fp8_k_nvfp4_v_routes_spec_decode_to_context() -> None:
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            parallel_drafting=False,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    builder.flashinfer_trtllm_api_decode_kernel = FlashInferDecodeKernel.TRTLLM_GEN
    builder.is_kvcache_fp8_k_nvfp4_v = True
    builder.use_dedicated_xqa = False

    builder._init_reorder_batch_threshold(
        1,
        supports_spec_as_decode=builder._supports_spec_as_decode(),
    )

    assert builder.reorder_batch_threshold == 1

    builder.is_kvcache_fp8_k_nvfp4_v = False
    builder._init_reorder_batch_threshold(
        1,
        supports_spec_as_decode=builder._supports_spec_as_decode(),
    )

    assert builder.reorder_batch_threshold == 4


def test_fp8_k_nvfp4_v_staged_prefill_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FP8_K_NVFP4_V_STAGED_PREFILL_MIN_Q", 8192)

    assert not use_staged_fp8_k_nvfp4_v_prefill(1)
    assert not use_staged_fp8_k_nvfp4_v_prefill(8191)
    assert use_staged_fp8_k_nvfp4_v_prefill(8192)

    monkeypatch.setattr(envs, "VLLM_FP8_K_NVFP4_V_STAGED_PREFILL_MIN_Q", 1)
    assert not use_staged_fp8_k_nvfp4_v_prefill(511)
    assert use_staged_fp8_k_nvfp4_v_prefill(512)


def test_fp8_k_nvfp4_v_disables_staging_for_cudagraph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    observed: dict[str, object] = {}

    def fake_build(**kwargs: object) -> object:
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(builder, "build", fake_build)
    common_attn_metadata = SimpleNamespace()
    result = builder.build_for_cudagraph_capture(common_attn_metadata)

    assert result is not None
    assert observed == {
        "common_prefix_len": 0,
        "common_attn_metadata": common_attn_metadata,
        "for_cudagraph_capture": True,
    }


def test_fp8_k_nvfp4_v_staging_requires_real_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FP8_K_NVFP4_V_STAGED_PREFILL_MIN_Q", 2)

    assert use_staged_fp8_k_nvfp4_v_prefill(8192)
    assert not use_staged_fp8_k_nvfp4_v_prefill(8192, is_dummy_run=True)
    assert not use_staged_fp8_k_nvfp4_v_prefill(
        8192, for_cudagraph_capture=True
    )


def test_fp8_k_nvfp4_v_staged_prefill_preserves_dummy_metadata() -> None:
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 4]),
        query_start_loc_cpu=torch.tensor([0, 2, 4]),
        seq_lens=torch.tensor([2, 2]),
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(4),
        is_dummy_run=True,
    )

    assert common.unpadded(num_actual_tokens=2, num_actual_reqs=1).is_dummy_run


def test_fp8_k_nvfp4_v_staged_prefill_v2_metadata_propagates_dummy_run() -> None:
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata

    class MetadataBuilder:
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
        ) -> CommonAttentionMetadata:
            assert common_prefix_len == 0
            return common_attn_metadata

    builder = MetadataBuilder()
    attn_group = SimpleNamespace(
        layer_names=["layer"],
        get_metadata_builder=lambda _: builder,
    )
    metadata = build_attn_metadata(
        attn_groups=[[attn_group]],
        num_reqs=1,
        num_tokens=2,
        query_start_loc_gpu=torch.tensor([0, 2]),
        query_start_loc_cpu=torch.tensor([0, 2]),
        max_query_len=2,
        seq_lens=torch.tensor([2]),
        max_seq_len=2,
        block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
        slot_mappings=torch.zeros((1, 2), dtype=torch.int64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        is_dummy_run=True,
    )

    assert metadata["layer"].is_dummy_run


@torch.inference_mode()
@pytest.mark.parametrize("head_size", [128, 256])
def test_fp8_k_nvfp4_v_staged_prefill_values(head_size: int) -> None:
    num_pages, num_heads, block_size = 5, 2, 8
    scale_dim = head_size // 16
    k_cache = torch.randn(
        num_pages,
        num_heads,
        block_size,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    fp4_codes = torch.randint(
        0,
        16,
        (num_pages, num_heads, block_size, head_size),
        dtype=torch.uint8,
        device="cuda",
    )
    v_cache = fp4_codes[..., 0::2] | (fp4_codes[..., 1::2] << 4)

    scale_choices = torch.tensor(
        [0.25, 0.5, 1.0, 2.0], dtype=torch.float32, device="cuda"
    )
    logical_scales = scale_choices[
        torch.randint(
            0,
            len(scale_choices),
            (num_pages, num_heads, block_size, scale_dim),
            device="cuda",
        )
    ].to(torch.float8_e4m3fn)
    physical_scales = torch.empty_like(logical_scales)
    scale_group = scale_dim // 4
    for token in range(block_size):
        for scale in range(scale_dim):
            swizzled_token = (token // 4) * 4 + scale // scale_group
            swizzled_scale = (scale % scale_group) * 4 + token % 4
            physical_scales[:, :, swizzled_token, swizzled_scale] = logical_scales[
                :, :, token, scale
            ]

    block_tables = torch.tensor([[0, 2], [4, -1]], dtype=torch.int32, device="cuda")
    staged_cache, staged_block_tables, uses_shared_paged_kv_idx = (
        trtllm_prefill_attn_fp8_k_nvfp4_v_unpack(
            k_cache, v_cache, physical_scales, block_tables
        )
    )
    e2m1 = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    staged_k_cache, staged_v_cache = staged_cache
    staged_block_table = torch.arange(
        1, 5, dtype=torch.int32, device="cuda"
    ).reshape(2, 2)
    staged_block_table[-1, -1] = -1
    torch.testing.assert_close(staged_block_tables, staged_block_table)
    assert uses_shared_paged_kv_idx
    for batch in range(block_tables.shape[0]):
        for page in range(block_tables.shape[1]):
            src_page = int(block_tables[batch, page].item())
            if src_page < 0:
                continue
            staged_page = int(staged_block_table[batch, page].item())
            expected_v = e2m1[fp4_codes[src_page].long()]
            expected_v *= logical_scales[src_page].float().repeat_interleave(16, -1)
            torch.testing.assert_close(staged_k_cache[staged_page], k_cache[src_page])
            torch.testing.assert_close(
                staged_v_cache[staged_page], expected_v.to(torch.float8_e4m3fn)
            )


@torch.inference_mode()
def test_fp8_k_nvfp4_v_store_rejects_fp16() -> None:
    page_size, num_heads, head_size = 64, 1, 128
    total_dim = head_size + head_size // 2 + head_size // 16
    packed_cache = torch.empty(
        1,
        page_size,
        num_heads,
        total_dim,
        dtype=torch.uint8,
        device="cuda",
    )
    key = torch.randn(
        1, num_heads, head_size, dtype=torch.float16, device="cuda"
    )
    value = torch.randn_like(key)
    slot_mapping = torch.zeros(1, dtype=torch.int64, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError):
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            packed_cache,
            packed_cache,
            slot_mapping,
            "fp8_k_nvfp4_v",
            scale,
            scale,
        )


def _make_test_mixed_cache(
    head_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_pages, page_size = 4, 64
    num_kv_heads = 4
    total_dim = head_size + head_size // 2 + head_size // 16
    packed_cache = torch.empty(
        num_pages,
        num_kv_heads,
        page_size,
        total_dim,
        dtype=torch.uint8,
        device="cuda",
    )
    key = torch.randn(
        num_pages * page_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn_like(key)
    slot_mapping = torch.arange(key.shape[0], dtype=torch.int64, device="cuda")
    k_scale = torch.tensor(0.5, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(0.75, dtype=torch.float32, device="cuda")

    packed_cache_nhd = packed_cache.permute(0, 2, 1, 3)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        packed_cache_nhd,
        packed_cache_nhd,
        slot_mapping,
        "fp8_k_nvfp4_v",
        k_scale,
        v_scale,
    )
    k_cache, v_cache, v_block_scales = fp8_k_nvfp4_v_cache_split_views(
        packed_cache, head_size
    )

    key_qdq = k_cache.bfloat16() * k_scale
    value_qdq = dequant_nvfp4_kv_cache(
        v_cache, v_block_scales, float(v_scale.item()), head_size, page_size
    )
    torch.testing.assert_close(
        key_qdq.permute(0, 2, 1, 3).reshape_as(key),
        key,
        atol=0.05,
        rtol=0.05,
    )
    return (
        k_cache,
        v_cache,
        v_block_scales,
        key_qdq,
        value_qdq,
        k_scale,
        v_scale,
    )


@torch.inference_mode()
@pytest.mark.parametrize("head_size", [128, 256])
def test_fp8_k_nvfp4_v_staged_prefill_context(head_size: int) -> None:
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    torch.manual_seed(0)
    num_kv_heads, group_size = 4, 4
    num_qo_heads = num_kv_heads * group_size
    (
        k_cache,
        v_cache,
        v_block_scales,
        key_qdq,
        value_qdq,
        k_scale,
        v_scale,
    ) = _make_test_mixed_cache(head_size)

    batch_size, pages_per_seq = 2, 2
    block_tables = torch.arange(
        batch_size * pages_per_seq, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, pages_per_seq)
    staged_cache, staged_block_tables, uses_shared_paged_kv_idx = (
        trtllm_prefill_attn_fp8_k_nvfp4_v_unpack(
            k_cache, v_cache, v_block_scales, block_tables
        )
    )
    # Staging preserves normalized FP8 K. The global k_scale is applied by
    # BMM1, just as it is for the native mixed cache path.
    staged_k_cache, staged_v_cache = staged_cache
    staged_block_table = torch.arange(
        1, 5, dtype=torch.int32, device="cuda"
    ).reshape(2, 2)
    staged_k_cache = staged_k_cache[1:]
    staged_v_cache = staged_v_cache[1:]
    torch.testing.assert_close(staged_block_tables, staged_block_table)
    assert uses_shared_paged_kv_idx
    torch.testing.assert_close(staged_k_cache.float() * k_scale, key_qdq)
    torch.testing.assert_close(
        staged_v_cache.float() * v_scale,
        value_qdq,
        atol=0.25,
        rtol=0.1,
    )
    seq_lens = torch.tensor([96, 128], dtype=torch.int32, device="cuda")
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    q_lens = torch.tensor([32, 32], dtype=torch.int32, device="cuda")
    cum_seq_lens_q = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
    cum_seq_lens_kv = torch.tensor([0, 96, 224], dtype=torch.int32, device="cuda")
    context_query = torch.randn(
        int(q_lens.sum().item()),
        num_qo_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    context_query_fp8, context_q_scale = _to_fp8(context_query)
    context_kwargs = dict(
        query=context_query_fp8,
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=int(q_lens.max().item()),
        max_kv_len=int(seq_lens.max().item()),
        bmm1_scale=(
            context_q_scale * float(k_scale.item()) / math.sqrt(head_size)
        ),
        bmm2_scale=float(v_scale.item()),
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        out_dtype=torch.bfloat16,
        kv_layout="HND",
    )
    staged_context = trtllm_batch_context_with_kv_cache(
        kv_cache=staged_cache,
        block_tables=staged_block_tables,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        **context_kwargs,
    )
    context_query_qdq = context_query_fp8.float() * context_q_scale
    context_ref = []
    for batch_idx, (q_len, seq_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist())
    ):
        page_ids = block_tables[batch_idx]
        key_seq = (
            key_qdq[page_ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_size)[:, :seq_len]
            .repeat_interleave(group_size, dim=0)
            .float()
        )
        value_seq = (
            value_qdq[page_ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_size)[:, :seq_len]
            .repeat_interleave(group_size, dim=0)
        )
        q_start = int(cum_seq_lens_q[batch_idx].item())
        query_seq = context_query_qdq[q_start : q_start + q_len]
        logits = torch.einsum("qhd,hnd->qhn", query_seq, key_seq)
        prefix_len = seq_len - q_len
        causal_mask = torch.arange(seq_len, device="cuda")[None, :] <= (
            prefix_len + torch.arange(q_len, device="cuda")[:, None]
        )
        probs = torch.softmax(
            (logits / math.sqrt(head_size)).masked_fill(
                ~causal_mask[:, None, :], float("-inf")
            ),
            dim=-1,
        )
        context_ref.append(torch.einsum("qhn,hnd->qhd", probs, value_seq))
    torch.testing.assert_close(
        staged_context,
        torch.cat(context_ref).to(torch.bfloat16),
        atol=0.25,
        rtol=0.1,
    )


@torch.inference_mode()
@pytest.mark.parametrize("head_size", [128, 256])
def test_fp8_k_nvfp4_v_store_and_native_decode(head_size: int) -> None:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    torch.manual_seed(0)
    num_kv_heads, group_size = 4, 4
    num_qo_heads = num_kv_heads * group_size
    (
        k_cache,
        v_cache,
        v_block_scales,
        key_qdq,
        value_qdq,
        k_scale,
        v_scale,
    ) = _make_test_mixed_cache(head_size)

    batch_size, pages_per_seq = 2, 2
    query = torch.randn(
        batch_size,
        num_qo_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_fp8, q_scale = _to_fp8(query)
    block_tables = torch.arange(
        batch_size * pages_per_seq, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, pages_per_seq)
    seq_lens = torch.tensor([96, 128], dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    output = trtllm_batch_decode_with_kv_cache(
        query_fp8,
        (k_cache, v_cache),
        workspace,
        block_tables,
        seq_lens,
        int(seq_lens.max().item()),
        bmm1_scale=q_scale * float(k_scale.item()) / math.sqrt(head_size),
        bmm2_scale=float(v_scale.item()),
        out_dtype=torch.bfloat16,
        backend="trtllm-gen",
        kv_layout="HND",
        kv_cache_sf=(None, v_block_scales),
    )

    output_ref = []
    query_qdq = query_fp8.float() * q_scale
    for batch_idx, seq_len in enumerate(seq_lens.tolist()):
        page_ids = block_tables[batch_idx]
        key_seq = (
            key_qdq[page_ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_size)[:, :seq_len]
            .repeat_interleave(group_size, dim=0)
            .float()
        )
        value_seq = (
            value_qdq[page_ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_size)[:, :seq_len]
            .repeat_interleave(group_size, dim=0)
        )
        logits = torch.einsum("hd,hnd->hn", query_qdq[batch_idx], key_seq)
        probs = torch.softmax(logits / math.sqrt(head_size), dim=-1)
        output_ref.append(torch.einsum("hn,hnd->hd", probs, value_seq))

    cosine = torch.nn.functional.cosine_similarity(
        output.float().reshape(-1), torch.stack(output_ref).reshape(-1), dim=0
    )
    assert cosine.item() > 0.99
