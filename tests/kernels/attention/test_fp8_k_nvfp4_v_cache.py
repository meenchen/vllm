# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import fp8_k_nvfp4_v_cache_split_views
from vllm.v1.attention.backends.flashinfer import (
    trtllm_prefill_attn_fp8_k_nvfp4_v_unpack,
)

FP8_DTYPE = current_platform.fp8_dtype()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FP8-K/NVFP4-V TRTLLM-Gen attention requires SM100",
)
@torch.inference_mode()
def test_fp8_k_nvfp4_v_store_and_native_decode() -> None:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_pages = 2
    page_size = 64
    num_kv_heads = 4
    group_size = 4
    num_qo_heads = num_kv_heads * group_size
    head_size = 128
    full_dim = head_size + head_size // 2 + head_size // 16

    physical_cache = torch.empty(
        (num_pages, 1, num_kv_heads, page_size, full_dim),
        dtype=torch.uint8,
        device=device,
    )
    logical_cache = physical_cache.permute(0, 1, 3, 2, 4)
    key = torch.randn(
        num_pages * page_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    slot_mapping = torch.arange(num_pages * page_size, dtype=torch.int64, device=device)
    k_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.75], dtype=torch.float32, device=device)

    packed_cache = logical_cache[:, 0]
    with pytest.raises(RuntimeError, match="alias one packed cache"):
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            packed_cache,
            torch.empty_like(packed_cache),
            slot_mapping,
            "fp8_k_nvfp4_v",
            k_scale,
            v_scale,
        )
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        packed_cache,
        packed_cache,
        slot_mapping,
        "fp8_k_nvfp4_v",
        k_scale,
        v_scale,
    )

    k_cache, v_cache, v_block_scales = fp8_k_nvfp4_v_cache_split_views(
        physical_cache[:, 0], head_size
    )
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        1, num_pages
    )
    normalized_cache, _ = trtllm_prefill_attn_fp8_k_nvfp4_v_unpack(
        k_cache,
        v_cache,
        v_block_scales,
        block_tables,
    )
    key_qdq = normalized_cache[1:, 0].float() * k_scale
    value_qdq = normalized_cache[1:, 1].float() * v_scale
    torch.testing.assert_close(
        key_qdq.permute(0, 2, 1, 3).reshape_as(key),
        key.float(),
        atol=0.05,
        rtol=0.05,
    )
    torch.testing.assert_close(
        value_qdq.permute(0, 2, 1, 3).reshape_as(value),
        value.float(),
        atol=1.5,
        rtol=0.5,
    )

    query = torch.randn(
        1,
        num_qo_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    query_fp8 = query.to(FP8_DTYPE)
    seq_lens = torch.tensor([num_pages * page_size], dtype=torch.int32, device=device)
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    output = trtllm_batch_decode_with_kv_cache(
        query=query_fp8,
        kv_cache=(k_cache, v_cache),
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=num_pages * page_size,
        bmm1_scale=float(k_scale.item()) / math.sqrt(head_size),
        bmm2_scale=float(v_scale.item()),
        out_dtype=torch.bfloat16,
        backend="trtllm-gen",
        kv_layout="HND",
        kv_cache_sf=(None, v_block_scales),
    )

    key_seq = (
        normalized_cache[1:, 0]
        .permute(1, 0, 2, 3)
        .reshape(num_kv_heads, -1, head_size)
        .repeat_interleave(group_size, dim=0)
        .float()
        * k_scale.item()
    )
    value_seq = (
        normalized_cache[1:, 1]
        .permute(1, 0, 2, 3)
        .reshape(num_kv_heads, -1, head_size)
        .repeat_interleave(group_size, dim=0)
        .float()
        * v_scale.item()
    )
    logits = torch.einsum("hd,hnd->hn", query_fp8[0].float(), key_seq)
    probs = torch.softmax(logits / math.sqrt(head_size), dim=-1)
    reference = torch.einsum("hn,hnd->hd", probs, value_seq)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().reshape(-1), reference.reshape(-1), dim=0
    )
    assert cosine.item() > 0.99


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FP8-K/NVFP4-V TRTLLM-Gen attention requires SM100",
)
@torch.inference_mode()
def test_fp8_k_nvfp4_v_prefill_unpack_masks_invalid_pages() -> None:
    device = torch.device("cuda")
    num_pages = 2
    num_kv_heads = 2
    block_size = 64
    head_size = 128
    full_dim = head_size + head_size // 2 + head_size // 16
    packed_cache = torch.zeros(
        (num_pages, num_kv_heads, block_size, full_dim),
        dtype=torch.uint8,
        device=device,
    )
    k_cache, v_cache, v_block_scales = fp8_k_nvfp4_v_cache_split_views(
        packed_cache, head_size
    )
    block_tables = torch.tensor(
        [[0, -1, num_pages + 17]], dtype=torch.int32, device=device
    )

    normalized_cache, _ = trtllm_prefill_attn_fp8_k_nvfp4_v_unpack(
        k_cache,
        v_cache,
        v_block_scales,
        block_tables,
    )

    torch.cuda.synchronize()
    assert normalized_cache.shape == (
        block_tables.numel() + 1,
        2,
        num_kv_heads,
        block_size,
        head_size,
    )
