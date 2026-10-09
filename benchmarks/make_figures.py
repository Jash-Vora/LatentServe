"""
Regenerate the README and docs charts from the saved result tables.

    python -m benchmarks.make_figures

Reads `docs/sweep_stage1_results.md` (the final single-GPU sweep, as printed by
`sweep_stage1 --report`) plus two numbers-bearing notes, and writes PNGs to
`docs/figures/`. No number is typed in here: every plotted value is parsed from
those files, so the charts change when the tables do.

README: decode_speedup, prefill_ttft, latency_vs_load, capacity_vs_sparsity
docs:   kernel_bandwidth (phase 12), prefix_caching (phase 13), sparse_vs_context (sweep note)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

LS, VLLM, SPARSE, INT8, GREY = "#2563eb", "#e08a00", "#16a34a", "#7c3aed", "#6b7280"
DECODE_COLS = ["ls-dense", "ls-sparse50", "ls-sparse37.5", "ls-int8", "vllm"]
CTX_LABEL = {2048: "2K", 8192: "8K", 16384: "16K", 32768: "32K"}
T4_PEAK_GBPS = 320          # NVIDIA T4 datasheet memory bandwidth


# ----------------------------------------------------------------------- parsing

def _cells(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _first_number(cell: str):
    m = re.match(r"-?\d+(?:\.\d+)?", cell.strip())
    return float(m.group()) if m else None


def _section(text: str, start: str, end: str | None) -> str:
    i = text.index(start)
    j = text.index(end, i + 1) if end else len(text)
    return text[i:j]


def parse_decode(text: str) -> dict:
    """{(batch, ctx, config): p50 ms, or None where the batch did not fit}."""
    block = _section(text, "## A. Decode step", "## B. Time to first token")
    header = next(_cells(l) for l in block.splitlines() if l.startswith("| batch"))
    assert header[2:] == DECODE_COLS, f"unexpected columns {header}"
    out = {}
    for line in block.splitlines():
        c = _cells(line) if line.lstrip().startswith("|") else []
        if len(c) == 7 and c[0].isdigit() and c[1].isdigit():
            for name, cell in zip(DECODE_COLS, c[2:]):
                out[(int(c[0]), int(c[1]), name)] = _first_number(cell) if "/" in cell else None
    return out


def parse_ttft(text: str) -> dict:
    """{(prompt tokens, config): p50 ms}."""
    block = _section(text, "## B. Time to first token", "## C. Serving workloads")
    names = ["ls-dense", "ls-int8", "vllm"]
    out = {}
    for line in block.splitlines():
        c = _cells(line) if line.lstrip().startswith("|") else []
        if len(c) == 4 and c[0].isdigit():
            for name, cell in zip(names, c[1:]):
                out[(int(c[0]), name)] = _first_number(cell)
    return out


def parse_load(text: str) -> dict:
    """{config: {"capacity": req/s, "points": [(req/s, ttft p50 ms, ttft p99 ms)]}}."""
    block = _section(text, "## C. Latency versus load", "## Drift sentinel")
    out, cur = {}, None
    for line in block.splitlines():
        m = re.match(r"\*\*(.+?)\*\* — capacity ([\d.]+) req/s", line)
        if m:
            cur = m.group(1)
            out[cur] = {"capacity": float(m.group(2)), "points": []}
            continue
        c = _cells(line) if line.lstrip().startswith("|") else []
        if cur and len(c) == 7 and c[0].endswith("%"):
            p50, p99 = (float(x) for x in c[4].split("/"))
            out[cur]["points"].append((float(c[1]), p50, p99))
    return out


def parse_kernel_bandwidth(phase12: str) -> dict:
    m = re.search(r"ran at (\d+) GB/s against (\d+) GB/s for a loads-only version\. "
                  r"A hand-written CUDA-core kernel reached (\d+) GB/s", phase12)
    assert m, "phase 12 outcome note changed: update parse_kernel_bandwidth"
    return {"triton": int(m.group(1)), "loads_only": int(m.group(2)), "cuda": int(m.group(3))}


def parse_prefix(phase13: str) -> dict:
    """{workload: (ttft p50 ms off, on)} for the fp16 rows."""
    out = {}
    for w, off, on in re.findall(r"\| (shared|chat|none) \| fp16 \| [\d.]+% \| ([\d.]+) -> ([\d.]+) ms", phase13):
        out[w] = (float(off), float(on))
    assert set(out) == {"shared", "chat", "none"}, out
    return out


# ------------------------------------------------------------------------ drawing

def _axes(title: str, xlabel: str, ylabel: str, size=(6.6, 4.0)):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_decode_speedup(dec: dict, out: Path) -> Path:
    fig, ax = _axes("Decode step: LatentServe vs vLLM", "batch size", "speedup  (vLLM step ÷ LatentServe step)")
    best = (0, None)
    for ctx, color in zip(CTX_LABEL, ["#93c5fd", "#60a5fa", "#2563eb", "#1e3a8a"]):
        pts = [(b, dec[(b, ctx, "vllm")] / dec[(b, ctx, "ls-dense")]) for b in (1, 4, 8, 16, 32)
               if dec.get((b, ctx, "vllm")) and dec.get((b, ctx, "ls-dense"))]
        ax.plot(*zip(*pts), marker="o", color=color, label=f"{CTX_LABEL[ctx]} context")
        for b, r in pts:
            if r > best[0]:
                best = (r, (b, ctx))
    ax.axhline(1, color=GREY, ls="--", lw=1)
    ax.text(32, 1.0, "tie", color=GREY, ha="right", va="bottom", fontsize=8)
    r, (b, ctx) = best
    ax.annotate(f"{r:.1f}×", (b, r), textcoords="offset points", xytext=(-6, 8), ha="right", fontsize=9, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_ylim(0.8, None)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, out, "decode_speedup.png")


def fig_prefill(ttft: dict, out: Path) -> Path:
    fig, ax = _axes("First-token time by prompt length (one request)", "prompt tokens", "first-token time (s)")
    lens = sorted({k[0] for k in ttft})
    for name, color, label in (("vllm", VLLM, "vLLM"), ("ls-dense", LS, "LatentServe")):
        ax.plot(lens, [ttft[(L, name)] / 1000 for L in lens], marker="o", color=color, label=label)
    for L in (lens[0], lens[-1]):
        ratio = ttft[(L, "vllm")] / ttft[(L, "ls-dense")]
        y = (ttft[(L, "vllm")] * ttft[(L, "ls-dense")]) ** 0.5 / 1000
        ax.annotate(f"{ratio:.1f}×", (L, y), textcoords="offset points", xytext=(-8 if L == lens[-1] else 8, 0),
                    ha="right" if L == lens[-1] else "left", va="center", fontsize=9, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lens)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v) // 1024}K" if v >= 1024 else f"{int(v)}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, out, "prefill_ttft.png")


def fig_load(load: dict, out: Path) -> Path:
    fig, ax = _axes("Latency under load (one T4)", "arrival rate (requests / s)", "first-token time (s)")
    for name, color, label in (("vllm", VLLM, "vLLM"), ("ls-dense", LS, "LatentServe")):
        pts = load[name]["points"]
        ax.plot([p[0] for p in pts], [p[2] / 1000 for p in pts], marker="o", ms=4, color=color, label=f"{label} p99")
        ax.plot([p[0] for p in pts], [p[1] / 1000 for p in pts], ls="--", lw=1.2, color=color, alpha=0.8, label=f"{label} p50")
        cap = load[name]["capacity"]
        ax.axvline(cap, color=color, ls=":", lw=1)
        ax.text(cap + (0.012 if name == "vllm" else -0.012), 12.5, f"saturates at {cap:.2f} req/s", color=color,
                rotation=90, va="center", ha="left" if name == "vllm" else "right", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.38, 1.0), ncol=2)
    fig.text(0.99, 0.01, "single run, 16–80 requests per point", ha="right", fontsize=7, color=GREY)
    return _save(fig, out, "latency_vs_load.png")


def fig_capacity(dec: dict, out: Path) -> Path:
    """Decode tokens/s: what doubling the batch with INT8 buys, against sparse attention."""
    fig, ax = _axes("Decode throughput: INT8 capacity vs sparse attention", "", "decode tokens / s", size=(6.6, 4.2))
    groups = [("16K context", 16384, 16, 16, 32), ("32K context", 32768, 8, 8, 16)]
    labels = ["fp16\ndense", "fp16 sparse\n37.5%", "INT8 dense\n(2× batch)"]
    colors = [LS, SPARSE, INT8]
    x, xs, tl = 0, [], []
    for title, ctx, b_dense, b_sparse, b_int8 in groups:
        vals = [b_dense * 1000 / dec[(b_dense, ctx, "ls-dense")], b_sparse * 1000 / dec[(b_sparse, ctx, "ls-sparse37.5")],
                b_int8 * 1000 / dec[(b_int8, ctx, "ls-int8")]]
        batches = [b_dense, b_sparse, b_int8]
        for i, (v, c) in enumerate(zip(vals, colors)):
            ax.bar(x + i, v, color=c, width=0.8)
            ax.text(x + i, v + 6, f"{v:.0f}", ha="center", fontsize=9)
            xs.append(x + i)
            tl.append(f"{labels[i]}\nbatch {batches[i]}")
        ax.text(x + 1, -0.33, title, transform=ax.get_xaxis_transform(), ha="center", fontsize=10, fontweight="bold")
        x += 4
    ax.set_xticks(xs)
    ax.set_xticklabels(tl, fontsize=8)
    ax.set_ylim(0, max(ax.get_ylim()) * 1.08)
    fig.subplots_adjust(bottom=0.3)
    path = out / "capacity_vs_sparsity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_kernel(bw: dict, out: Path) -> Path:
    fig, ax = _axes("Decode attention bandwidth (batch 16, 8K)", "GB/s", "", size=(6.6, 3.2))
    rows = [("Triton kernel", bw["triton"], VLLM), ("CUDA-core kernel", bw["cuda"], LS),
            ("loads only (ceiling)", bw["loads_only"], GREY), ("T4 peak (spec)", T4_PEAK_GBPS, "#d1d5db")]
    for i, (name, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.62)
        ax.text(v + 4, i, f"{v}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, T4_PEAK_GBPS * 1.1)
    ax.grid(axis="y", alpha=0)
    return _save(fig, out, "kernel_bandwidth.png")


def fig_prefix(pfx: dict, out: Path) -> Path:
    fig, ax = _axes("Prefix caching: first-token time", "", "first-token time, p50 (ms)")
    names = [("shared", "shared system\nprompt"), ("chat", "multi-turn\nchat"), ("none", "nothing\nshared")]
    for i, (key, label) in enumerate(names):
        off, on = pfx[key]
        ax.bar(i - 0.2, off, 0.38, color=GREY, label="prefix caching off" if i == 0 else None)
        ax.bar(i + 0.2, on, 0.38, color=LS, label="on" if i == 0 else None)
        ax.text(i - 0.2, off + 6, f"{off:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, on + 6, f"{on:.0f}\n({(on / off - 1) * 100:+.0f}%)", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[1] for n in names])
    ax.set_ylim(0, max(max(v) for v in pfx.values()) * 1.35)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="x", alpha=0)
    return _save(fig, out, "prefix_caching.png")


def fig_sparse_context(dec: dict, out: Path) -> Path:
    fig, ax = _axes("Sparse attention pays off at long context (batch 8)", "context length", "decode step (ms)")
    ctxs = sorted(CTX_LABEL)
    for name, color, label in (("ls-dense", LS, "dense"), ("ls-sparse50", "#86efac", "sparse 50%"),
                               ("ls-sparse37.5", SPARSE, "sparse 37.5%")):
        ax.plot(ctxs, [dec[(8, c, name)] for c in ctxs], marker="o", color=color, label=label)
    d, s = dec[(8, 32768, "ls-dense")], dec[(8, 32768, "ls-sparse37.5")]
    ax.annotate(f"{d / s:.2f}× faster", (32768, s), textcoords="offset points", xytext=(8, -30), ha="right", fontsize=9, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctxs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: CTX_LABEL.get(int(v), "")))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, out, "sparse_vs_context.png")


def make_all(results: Path, out: Path, phase12: Path, phase13: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    text = results.read_text()
    dec, ttft, load = parse_decode(text), parse_ttft(text), parse_load(text)
    return [fig_decode_speedup(dec, out), fig_prefill(ttft, out), fig_load(load, out), fig_capacity(dec, out),
            fig_kernel(parse_kernel_bandwidth(phase12.read_text()), out),
            fig_prefix(parse_prefix(phase13.read_text()), out), fig_sparse_context(dec, out)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", default="docs/sweep_stage1_results.md")
    p.add_argument("--phase12", default="docs/phase12_kernel_findings.md")
    p.add_argument("--phase13", default="docs/phase13_prefix.md")
    p.add_argument("--out", default="docs/figures")
    a = p.parse_args()
    for path in make_all(Path(a.results), Path(a.out), Path(a.phase12), Path(a.phase13)):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
