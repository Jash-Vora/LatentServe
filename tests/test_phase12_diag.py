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


def test_sass_parser_counts_spills_loads_and_tensor_cores():
    from benchmarks.runners.phase12_diag import parse_sass

    sass = """
        .text._paged_decode_tiled:
        /*0000*/                   MOV R1, c[0x0][0x28] ;          /* 0x00000a0000017a02 */
        /*0010*/                   LDG.E.128.SYS R4, [R2] ;        /* 0x0000000402047381 */
        /*0020*/              @!P0 LDG.E.64.SYS R8, [R6] ;         /* 0x0000000406087381 */
        /*0030*/                   LDG.E.U16.SYS R9, [R6+0x2] ;    /* 0x0000020406097381 */
        /*0040*/                   STL.64 [R1+0x10], R4 ;          /* 0x0000100401007387 */
        /*0050*/                   LDL.64 R4, [R1+0x10] ;          /* 0x0000100001047983 */
        /*0060*/                   HMMA.1688.F32 R12, R16, R18, R12 ; /* 0x000000121010723c */
        /*0070*/               @P1 BAR.SYNC 0x0 ;                  /* 0x0000000000007b1d */
        /*0080*/                   MUFU.EX2 R3, R3 ;               /* 0x0000000300037308 */
    .L_x_0:
    """
    c = parse_sass(sass)
    assert c["total"] == 9
    assert c["LDG"] == 3 and c["LDG 128"] == 1 and c["LDG 64"] == 1 and c["LDG 32 or less"] == 1
    assert c["STL"] == 1 and c["LDL"] == 1                 # spills, made concrete
    assert c["HMMA"] == 1 and c["BAR"] == 1 and c["MUFU"] == 1