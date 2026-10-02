"""The diagnostic's own arithmetic: occupancy from compiler resources."""

from benchmarks.runners.phase12_diag import occupancy


def test_registers_bound_occupancy():
    # 128 regs x 32 threads = 4096 per warp -> 16 warps on 65,536 registers.
    occ, bound = occupancy(regs=128, smem=0, warps=4)
    assert occ == 0.5 and bound == "registers"


def test_heavy_register_use_quarters_it():
    occ, bound = occupancy(regs=255, smem=0, warps=4)
    assert occ == 0.25 and bound == "registers"


def test_shared_memory_can_be_the_limit():
    occ, bound = occupancy(regs=32, smem=40 * 1024, warps=4)
    assert bound == "shared memory" and occ == 4 / 32


def test_unknown_resources_fall_back_to_slot_limits():
    occ, bound = occupancy(regs=0, smem=0, warps=4)
    assert occ == 1.0