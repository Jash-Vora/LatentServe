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


import pytest  # noqa: E402

from benchmarks.runners.vllm_backend_probe import reported_graph_mode  # noqa: E402
from comparisons.vllm.runner import parse_variant, variant_label  # noqa: E402


def test_variants_parse_into_a_backend_and_engine_args():
    assert parse_variant("auto") == (None, {})
    assert parse_variant("FLASHINFER") == ("FLASHINFER", {})
    assert parse_variant("auto+full_graphs+async") == (
        None, {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
               "async_scheduling": True})
    assert parse_variant("FLASHINFER+async") == ("FLASHINFER", {"async_scheduling": True})


def test_contradictory_or_unknown_knobs_are_refused():
    with pytest.raises(ValueError, match="conflicts"):
        parse_variant("auto+full_graphs+piecewise")
    with pytest.raises(ValueError, match="unknown knob"):
        parse_variant("auto+turbo")


def test_labels_keep_the_default_arm_named_vllm():
    assert variant_label("auto") == "vllm"
    assert variant_label("FLASHINFER") == "vllm_flashinfer"
    assert variant_label("auto+full_graphs+async") == "vllm_auto_full_graphs_async"


@pytest.mark.parametrize("line, mode", [
    ("compilation_config={'level': 3, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}",
     "FULL_AND_PIECEWISE"),
    ('"cudagraph_mode": "PIECEWISE"', "PIECEWISE"),
    ("cudagraph_mode=FULL_DECODE_ONLY", "FULL_DECODE_ONLY"),
    ("no graph settings printed", None),
])
def test_reads_the_graph_mode_vllm_printed(line, mode):
    assert reported_graph_mode(line) == mode