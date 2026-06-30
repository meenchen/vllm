# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.utils.torch_utils import (
    common_broadcastable_dtype,
    current_stream,
    fp8_k_nvfp4_v_cache_split_views,
    is_lossless_cast,
)


@pytest.mark.parametrize(
    ("src_dtype", "tgt_dtype", "expected_result"),
    [
        # Different precision_levels
        (torch.bool, torch.int8, True),
        (torch.bool, torch.float16, True),
        (torch.bool, torch.complex32, True),
        (torch.int64, torch.bool, False),
        (torch.int64, torch.float16, True),
        (torch.int64, torch.complex32, True),
        (torch.float64, torch.bool, False),
        (torch.float64, torch.int8, False),
        (torch.float64, torch.complex32, True),
        (torch.complex128, torch.bool, False),
        (torch.complex128, torch.int8, False),
        (torch.complex128, torch.float16, False),
        # precision_level=0
        (torch.bool, torch.bool, True),
        # precision_level=1
        (torch.int8, torch.int16, True),
        (torch.int16, torch.int8, False),
        (torch.uint8, torch.int8, False),
        (torch.int8, torch.uint8, False),
        # precision_level=2
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float16, False),
        (torch.bfloat16, torch.float32, True),
        (torch.float32, torch.bfloat16, False),
        # precision_level=3
        (torch.complex32, torch.complex64, True),
        (torch.complex64, torch.complex32, False),
    ],
)
def test_is_lossless_cast(src_dtype, tgt_dtype, expected_result):
    assert is_lossless_cast(src_dtype, tgt_dtype) == expected_result


@pytest.mark.parametrize(
    ("dtypes", "expected_result"),
    [
        ([torch.bool], torch.bool),
        ([torch.bool, torch.int8], torch.int8),
        ([torch.bool, torch.int8, torch.float16], torch.float16),
        ([torch.bool, torch.int8, torch.float16, torch.complex32], torch.complex32),  # noqa: E501
    ],
)
def test_common_broadcastable_dtype(dtypes, expected_result):
    assert common_broadcastable_dtype(dtypes) == expected_result


def _test_stream_thread(main_expected_stream: torch.cuda.Stream):
    import threading

    child_stream = torch.cuda.Stream()
    thread_stream_ready = threading.Event()
    thread_can_exit = threading.Event()

    def child_thread_func():
        with torch.cuda.stream(child_stream):
            thread_stream_ready.set()
            thread_can_exit.wait(timeout=10)

    child_thread = threading.Thread(target=child_thread_func)
    child_thread.start()

    try:
        assert thread_stream_ready.wait(timeout=5), (
            "Child thread failed to enter stream context in time"
        )

        main_current_stream = current_stream()

        assert main_current_stream != child_stream, (
            "Main thread's current_stream was contaminated by child thread"
        )
        assert main_current_stream == main_expected_stream, (
            f"Main thread's stream changed unexpectedly. "
            f"Expected {main_expected_stream}, got {main_current_stream}"
        )

        thread_can_exit.set()

    finally:
        child_thread.join(timeout=5)
        if child_thread.is_alive():
            pytest.fail("Child thread failed to exit properly")


def test_current_stream_multithread():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    main_dedicated_stream = current_stream()

    assert main_dedicated_stream.cuda_stream != 0, (
        "ROCm/CUDA should create a dedicated stream, not use default stream (0x0)"
    )

    main_stream_again = current_stream()
    assert main_stream_again == main_dedicated_stream, (
        "Multiple calls to current_stream should return the same dedicated stream"
    )

    _test_stream_thread(main_dedicated_stream)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_fp8_k_nvfp4_v_cache_split_views(layout: str):
    num_pages, block_size, num_heads, head_size = 2, 16, 4, 128
    data_dim = head_size // 2
    scale_dim = head_size // 16
    total_dim = head_size + data_dim + scale_dim

    if layout == "NHD":
        physical = torch.zeros(
            num_pages, block_size, num_heads, total_dim, dtype=torch.uint8
        )
        kv_cache = physical
        expected_k_strides = (
            block_size * num_heads * total_dim,
            num_heads * head_size,
            head_size,
            1,
        )
    else:
        physical = torch.zeros(
            num_pages, num_heads, block_size, total_dim, dtype=torch.uint8
        )
        kv_cache = physical.permute(0, 2, 1, 3)
        expected_k_strides = (
            block_size * num_heads * total_dim,
            head_size,
            block_size * head_size,
            1,
        )

    k_cache, v_data, v_scale = fp8_k_nvfp4_v_cache_split_views(kv_cache, head_size)

    assert k_cache.shape == (num_pages, block_size, num_heads, head_size)
    assert v_data.shape == (num_pages, block_size, num_heads, data_dim)
    assert v_scale.shape == (num_pages, block_size, num_heads, scale_dim)
    assert k_cache.dtype == torch.float8_e4m3fn
    assert v_data.dtype == torch.uint8
    assert v_scale.dtype == torch.float8_e4m3fn
    assert k_cache.stride() == expected_k_strides

    values_per_page = block_size * num_heads
    assert v_data.storage_offset() == values_per_page * head_size
    assert v_scale.storage_offset() == values_per_page * (head_size + data_dim)
    k_cache.view(torch.uint8).fill_(0x11)
    v_data.fill_(0x22)
    v_scale.view(torch.uint8).fill_(0x33)
    pages = physical.flatten(1)
    k_end = values_per_page * head_size
    v_end = k_end + values_per_page * data_dim
    scale_end = v_end + values_per_page * scale_dim
    assert torch.all(pages[:, :k_end] == 0x11)
    assert torch.all(pages[:, k_end:v_end] == 0x22)
    assert torch.all(pages[:, v_end:scale_end] == 0x33)
    assert scale_end == pages.shape[1]


def test_fp8_k_nvfp4_v_cache_split_views_rejects_invalid_layout():
    with pytest.raises(ValueError, match="4D uint8"):
        fp8_k_nvfp4_v_cache_split_views(torch.empty(2, 16, 4, 200), 128)

    with pytest.raises(ValueError, match="last dim"):
        fp8_k_nvfp4_v_cache_split_views(
            torch.empty(2, 16, 4, 201, dtype=torch.uint8), 128
        )

    with pytest.raises(ValueError, match="contiguous last dimension"):
        fp8_k_nvfp4_v_cache_split_views(
            torch.empty(2, 16, 4, 200, 2, dtype=torch.uint8)[..., 0], 128
        )
