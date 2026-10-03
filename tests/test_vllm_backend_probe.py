"""The probe's reading of vLLM's log — what it reports as having run."""

from benchmarks.runners.vllm_backend_probe import failure_reason, reported_backends

LOG = """
INFO 10-03 08:00:00 [cuda.py:372] Using Triton Attention backend on V1 engine.
WARNING 10-03 08:00:00 [cuda.py:300] Cannot use FlashAttention backend for Volta and Turing GPUs.
INFO 10-03 08:00:01 [cuda.py:380] Using FlashInfer backend.
INFO 10-03 08:00:02 [selector.py:90] Using AttentionBackendEnum.TRITON_ATTN backend.
INFO 10-03 08:00:03 [cuda.py:372] Using Triton Attention backend on V1 engine.
"""


def test_reports_each_backend_vllm_says_it_used_once():
    assert reported_backends(LOG) == ["Triton", "FlashInfer", "AttentionBackendEnum.TRITON_ATTN"]


def test_a_refusal_is_not_a_report():
    assert reported_backends("Cannot use FlashAttention backend for Turing") == []


def test_failure_reason_is_the_last_error_line():
    log = "loading...\nValueError: Invalid attention backend: 'NOPE'\nexiting"
    assert failure_reason(log).startswith("ValueError: Invalid attention backend")