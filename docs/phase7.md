# Phase 7 — KV Representation & Latent Compression Study

Reframed from the methodology doc's "MLA-Inspired Latent KV Attention".
The question is not "does MLA help" but:

> **How much can Qwen's existing GQA KV state be compressed, and when
> does an MLA-style latent representation beat the cheaper
> alternatives?**

MLA stays the centrepiece. Low-rank, quantization and token selection are
**controls and alternative explanations**, not competitors: if MLA wins,
this design says why; if INT8 matches it for a fraction of the
complexity, that is the finding.

## The constraint that shapes everything

Qwen2.5-1.5B is already GQA: 2 KV heads x 128 dims, K and V = **512
numbers per token per layer**. DeepSeek's headline MLA numbers are stated
against MHA, where there is 10-50x to win. Here the baseline is already
compressed, and a latent must store `latent_dim + rope_dim`, so it only
saves anything below `latent_dim = 448`.

Three of the five values in the methodology doc's Phase 10 sweep (512,
768, 1024) would make the cache **larger** than the baseline. Re-centre
on roughly 96 / 128 / 192 / 256 / 384.

## Order of work

    7.0 baseline (freeze the Phase 6 runtime)
    7.1 spectral analysis          <- decides whether the rest is worth building
    7.7 INT8 control               <- moved early: it is the bar to clear
    7.2 uniform low-rank
    7.3 layer-adaptive rank
    7.4 MLA-inspired latent KV
        offline token-selective probe (runtime support is Phase 14)
    7.8 regime map

Two changes from the draft plan, both from what earlier phases measured:

* **Layer-adaptive, not head-adaptive (7.3).** Qwen has *two* KV heads,
  so per-head budgeting has two knobs and is close to vacuous. It has 28
  layers, and layer-wise variation is where the budget can actually move.
  Per-head spectra are still collected — they are free, and K-vs-V per
  head is informative — but the adaptive experiment allocates across
  layers.
* **INT8 first, not last (7.7).** It is a day of work for a hard 2x with
  no reconstruction compute. Running it last risks discovering in week 12
  that the cheap control matched the expensive method. The draft plan
  names exactly this fear and then schedules it where the fear comes
  true.

## 7.1 — what it measures and why

`compression/spectra.py`, `benchmarks/runners/phase7_spectra.py`.

```bash
python -m benchmarks.runners.phase7_spectra --context-length 8192
python -m benchmarks.runners.phase7_spectra --context-length 8192 --random-tokens   # control
```

CPU, minutes, no benchmark. Reports, per layer, the rank needed for
90/95/99/99.9% of the energy for:

| spectrum | why |
| --- | --- |
| `kv_joint` | the vector an MLA latent compresses: one shared vector per token per layer, across both heads and both of K and V |
| `k_pre_rope` / `k_post_rope` | the RoPE trap, below |
| `v` | no positional encoding, so expected to compress more readily |
| `*_head0/1` | K-vs-V per head; feeds the (weak) head-adaptive option |
| `w_k`, `w_v`, `w_kv_joint` | data-free weight spectra, as contrast |

### Gram accumulation, not stored activations

The squared singular values of A (N x d) are the eigenvalues of A^T A,
which is d x d whatever N is. So the analysis streams: 32K tokens cost
the same 2 MB as 32. Storing activations instead would be
28 x 32K x 512 x 4 B = 1.8 GB and would cap the sample size for nothing.
Accumulated in float64 — fp16 sums over tens of thousands of tokens lose
the small singular values, which are the entire question.

### The activation spectrum *is* the activation-aware weight analysis

For K = H W_K, the activation Gram is `W_K^T (H^T H) W_K` — the weight
matrix seen through the input covariance. A direction the weights treat
as important but the data never excites is free to discard; one the
weights barely touch but the data hammers is not. This matters because it
means no separate whitening step is needed, which is the expensive part
of methods like SVD-LLM.

### The RoPE trap

K is rotated by position *after* projection, and the rotation mixes
dimensions differently at every position. Taking the spectrum of
post-RoPE K therefore measures a structure deliberately smeared across
positions, and will make K look less compressible than it is.

Both are measured. Compressing pre-RoPE — cache the latent, reconstruct
K, rotate at the reconstructed position — is the structure MLA actually
uses, and the gap between the two spectra is a direct measurement of why
DeepSeek puts the positional path outside the compressed one.

### Calibration data is a controlled variable

Activation spectra depend on the input distribution, so **random token
ids are not a valid sample**: they are out of distribution and say
nothing about real serving. Phases 1-6 used random ids happily because
they measured time, and time does not care what the tokens mean. This
does.

Default is embedded natural-language passages in mixed registers;
`--text-file` takes a real corpus and is better. `--random-tokens` runs
the invalid version deliberately, and the gap between them is worth
reporting, because it shows how much a compressibility claim depends on
its calibration set.

### The decision this makes

If the joint KV needs ~128 of 512 dims for 99% energy, a latent
representation has ~2.7x of room and 7.2-7.4 are worth building. If it
needs 400+, there is nothing to compress, and that negative result —
with spectra behind it — is itself the finding, arrived at in an hour
rather than six weeks.

## 7.1 results (Qwen2.5-1.5B, 8192 tokens)

| spectrum | dim | r99 | r99/dim | effective rank |
| --- | ---: | ---: | ---: | ---: |
| `k_pre_rope` | 256 | 83 | 0.33 | **5.6** |
| `k_post_rope` | 256 | 157 | 0.61 | 17.3 |
| `v` | 256 | 196 | 0.77 | 72.2 |
| `kv_joint` | 512 | 165 | 0.32 | 14.6 |
| `w_kv_joint` (weights) | 512 | 433 | 0.85 | 316.3 |

**V is the bottleneck, not K.** Pre-RoPE K needs a third of its
dimensions; V needs three quarters. The single shared latent MLA uses has
to serve both, so V sets the budget.

**RoPE costs 1.89x.** K needs 83 dims pre-rotation and 157 post. Rotation
mixes dimensions differently at every position, smearing the structure —
compress pre-RoPE and rotate on reconstruction, and the decoupled
positional path stops being an architectural curiosity and becomes a
measured 1.9x.

**The data-aware analysis was worth doing.** The weight spectra say 433
of 512 (barely compressible); the activation spectra say 165. A
weights-only SVD would have concluded there was nothing here.

**Calibration barely mattered — my warning was overstated.** Real text vs
random ids: joint r99 165 vs 160, pre-RoPE K 83 vs 73. The spectra are
dominated by model structure rather than input distribution. Worth
reporting, since it is evidence *against* a caveat this document
previously stated strongly. (Worst-layer identity does move — layer 22 on
text, layer 4 on random — so per-layer budgets should use real text.)

### Why 2.24x should not be quoted yet

`kv_joint` r99 = 165 implies a 2.24x smaller cache. Do not report that
number. Retained energy is **norm-weighted**, and the giveaway is that
joint (165) is *lower* than V alone (196): reconstructing both to 99%
cannot be cheaper than reconstructing the harder one. K's activations
simply dominate the joint magnitude.

The effective ranks say why. `k_pre_rope` has r99 = 83 but effective rank
**5.6** — a handful of directions carrying nearly all the magnitude,
i.e. the massive-activation phenomenon. The per-layer plot shows the
extreme case: layers 0, 1 and 15 register r99 near zero, which does not
mean their KV is free to discard.

So the runner now also reports, for each candidate rank, the **relative
reconstruction error of the K block and the V block separately** under a
rank-r projection of the joint vector. Computable from the Gram alone.
`test_energy_threshold_hides_a_badly_reconstructed_block` pins the
failure down: a synthetic case where 99% of joint energy is reached at
rank 3 while the second block is reconstructed with 99.96% error.

**The usable rank is set by whichever block reconstructs worse**, and
even that is a proxy — the deciding metric is logit KL divergence after
truncation, which is 7.2's job.

## 7.1 reconstruction table — the honest answer

| rank | cache ratio | K rel. error | V rel. error |
| ---: | ---: | ---: | ---: |
| 128 | 2.67x | 0.100 | 0.304 |
| 192 | 2.00x | 0.062 | 0.190 |
| 256 | 1.60x | 0.042 | 0.113 |
| 384 | 1.14x | 0.017 | 0.030 |

**2.24x was an illusion.** V's error is 3-5x K's at every rank; around
rank 165, where the energy curve claimed 99%, V carries ~25% relative
error. The energy threshold was measuring K's magnitude and calling it
success. Real compression at 10% V error is ~1.6x.

Separate K/V latents do not rescue it: each block at 10% error needs
83 + 196 = 279 dims plus 64 positional, about 1.49x, against 1.60x for
the joint version at 11.3%. V is the binding constraint and no
arrangement escapes it.

**Calibration is a non-issue, definitively.** V error at rank 48 is 0.554
on text and 0.546 on random ids; joint r99 165 vs 160. Two independent
measurements say these spectra are set by model structure, not input
distribution. This document previously warned the opposite; that warning
was wrong.

## 7.2 — does reconstruction error become output damage?

`compression/truncation.py`, `benchmarks/runners/phase7_truncation.py`.

```bash
python -m benchmarks.runners.phase7_truncation --context-length 4096
```

Reconstruction error cannot answer the question that matters. Attention
is a softmax over dot products and may be far more or far less forgiving
than a Frobenius norm suggests, so the compression is **simulated inside
the forward pass** and scored by logit KL divergence and top-1 flip rate
against the unmodified model.

Simulation, not implementation: every method in this phase — low-rank,
adaptive rank, MLA, INT8 — has the same effect on the arithmetic, namely
that attention sees a lossy KV. What differs between them is storage and
speed, not output. So the quality question is settled for all of them
before any of them exists.

`k_proj` and `v_proj` are separate modules, so a hook on the first cannot
see V. Both wrappers therefore recompute both projections from the shared
hidden state, form the joint vector, project, and return their own slice.
Wasteful and irrelevant — nothing here is timed.

### The comparison the phase turns on

INT8 gives exactly 2x with **no reconstruction compute**, which was
already the term most likely to make MLA-2 net-slower on a T4. So the
question is not "is low-rank lossy" but "is low-rank at 2.0x better than
INT8 at 2.0x". If not, a latent representation needs a justification
other than memory saving.

Three INT8 granularities are swept, and 7.1's spectra predict which
should win: K has effective rank 5.6 (a few channels carrying enormous
magnitude, so a shared per-tensor scale starves the rest) and wants
**per-channel**; V's energy is spread evenly across 72 effective
dimensions and wants **per-token**. That is the KIVI arrangement,
arrived at independently from the spectra — a nice mutual check.

### One caveat to carry into the report

Subspaces are fitted on the same tokens the quality is measured on. That
is the *optimistic* case — an upper bound on what a fitted low-rank
method can achieve. 7.1 showed the spectra barely move between real text
and random ids, so the generalisation gap should be small, but
"should be" is not "measured". A held-out calibration set is the honest
version and is one flag away.

## Measurement protocol for 7.2 onward

From Phase 6's decode decomposition at 8K / batch 4: weights 28.1 ms,
attention read 8.7 ms, gather 17.3 ms. **Every method in this phase
touches only the second and third terms.** So:

* measure **differenced decode cost** (`--output-lengths 128 256`), never
  end-to-end, which at 8K is 46% prefill and would bury the effect;
* **batch 4-8, context 4K-16K** — at batch 1 Phase 2 found halving the
  cache made decode *slower*, because 2 KV heads on 40 SMs is an
  occupancy problem, not a bandwidth one;
* quality as **KL divergence from the fp16 baseline plus top-1
  agreement** — no corpus, no download, and it isolates compression
  damage from everything else. Perplexity and retrieval arrive in
  Phase 15.

One interaction to plan for: MLA and the Phase 11 kernel attack the same
bytes. If Phase 11 removes the gather first, MLA's decode win roughly
halves.

## Gate 7 checklist

- [ ] `pytest tests/test_phase7_spectra.py` green
- [ ] spectra run on real text at >= 8K tokens, and on random ids as the
      control, with the gap reported
- [ ] per-layer compressibility map produced (the 7.3 budget)
- [ ] pre- vs post-RoPE K gap quantified
- [ ] break-even stated explicitly against the 512-number GQA baseline
- [ ] go/no-go on 7.2-7.4 recorded in writing, with the number behind it