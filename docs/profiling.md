# Profiling Methodology

Nsight Compute is refused in the container these runs used (`ERR_NVGPUCTRPERM`), so kernels were not profiled with `ncu`. Instead the decode kernel was analysed from its compiler resource counts, ablation kernels and a census of its compiled machine code: see `benchmarks/runners/phase12_diag.py` and `phase12_profile.py`, with the findings in `phase12_kernel_findings.md`. Kernel launches per decode step are counted with the PyTorch profiler in `phase16_kernel_census.py`.

If Nsight is available on your machine, reports go in `profiling/nsight_systems/` and `profiling/nsight_compute/`. They are large binaries and gitignored: commit the commands, not the files.
