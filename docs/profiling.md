# Profiling Methodology

TODO — write once Phase 12 lands. Should document:
- exact nsys / ncu commands used for each profile (so results are reproducible)
- where .nsys-rep / .ncu-rep files are stored (profiling/nsight_systems/, profiling/nsight_compute/ — gitignored, large binaries; commit the commands, not the files, unless you set up Git LFS)
- how to read the specific metrics this project cares about: achieved occupancy,
  DRAM bandwidth utilization, L2 hit rate, warp stalls, tensor-core utilization
