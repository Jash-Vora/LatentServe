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


from benchmarks.runners.vllm_backend_probe import same_backend  # noqa: E402
from comparisons.vllm.runner import construct_engine  # noqa: E402


@pytest.mark.parametrize("requested, reported, same", [
    ("TRITON_ATTN", ["Triton"], True),                       # 'Using Triton Attention backend'
    ("TRITON_ATTN", ["AttentionBackendEnum.TRITON_ATTN"], True),
    ("FLASHINFER", ["FlashInfer"], True),
    ("FLASHINFER", ["TRITON_ATTN"], False),                  # what the first probe saw
    ("FLASH_ATTN", ["FlashInfer"], False),                   # a substring test says True
    ("TORCH_SDPA", ["TORCH_SDPA"], True),
])
def test_backend_names_compare_after_normalising(requested, reported, same):
    assert same_backend(requested, reported) is same


def _fake_llm(unknown=(), fail=None):
    """An engine that rejects the named options the way vLLM's EngineArgs does."""
    class FakeLLM:
        def __init__(self, **kw):
            for key in unknown:
                if key in kw:
                    raise TypeError(f"EngineArgs.__init__() got an unexpected keyword argument '{key}'")
            if fail:
                raise fail
            self.kw = kw
    return FakeLLM


def test_first_route_the_engine_accepts_is_used():
    llm, info = construct_engine(_fake_llm(), {"model": "m"}, False, "FLASHINFER")
    assert info["attention_backend_route"] == "attention_backend"
    assert llm.kw["attention_backend"] == "FLASHINFER" and info["attention_backend_rejected"] == {}


def test_unknown_routes_are_skipped_and_recorded():
    llm, info = construct_engine(_fake_llm(unknown=("attention_backend",)), {"model": "m"},
                                 False, "FLASHINFER")
    assert info["attention_backend_route"] == "attention_config"
    assert llm.kw["attention_config"] == {"backend": "FLASHINFER"}
    assert "attention_backend" in info["attention_backend_rejected"]
    _, info = construct_engine(_fake_llm(unknown=("attention_backend", "attention_config")),
                               {"model": "m"}, False, "FLASHINFER")
    assert info["attention_backend_route"] == "environment"


def test_an_unrelated_unknown_argument_is_never_dropped():
    """The old fallback caught any TypeError and retried with fewer arguments:
    an unknown knob would vanish and the run would carry on without it."""
    with pytest.raises(TypeError, match="async_scheduling"):
        construct_engine(_fake_llm(unknown=("async_scheduling",)),
                         {"model": "m", "async_scheduling": True}, False, None)


def test_prefix_caching_fallback_fires_only_for_its_own_argument():
    _, info = construct_engine(_fake_llm(unknown=("enable_prefix_caching",)), {"model": "m"},
                               False, None)
    assert info["prefix_caching_control"] == "unsupported_by_this_version"


def test_a_backend_that_fails_to_start_is_an_error_not_a_fallback():
    with pytest.raises(RuntimeError, match="not supported"):
        construct_engine(_fake_llm(fail=RuntimeError("FLASHINFER not supported on sm_75")),
                         {"model": "m"}, False, "FLASHINFER")