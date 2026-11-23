"""plots.py - standalone figures for a train.py run.

Reads runs/<name>/log.jsonl and any samples_*.png. Does not import train.py,
does not need torch. Forces the Agg backend so no GUI or Qt backend is needed.

    pip install matplotlib pillow
    python plots.py --run runs/dcgan128_sn
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError:
    raise SystemExit("matplotlib is missing. Run:  pip install matplotlib pillow")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------ input
def load_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "it" in r and "dacc" in r:
            rows.append(r)
    if not rows:
        raise SystemExit(f"no usable records in {path}")
    rows.sort(key=lambda r: r["it"])
    return rows


def first_sustained(rows, pred, window):
    """First iteration of a stretch lasting at least `window` iterations."""
    start = None
    for r in rows:
        if pred(r):
            if start is None:
                start = r["it"]
            if r["it"] - start >= window:
                return start
        else:
            start = None
    return None


def balance_report(rows: list[dict], window=2000) -> list[str]:
    last_it = rows[-1]["it"]
    tail = [r for r in rows if r["it"] >= 0.8 * last_it]
    tail_acc = float(np.mean([r["dacc"] for r in tail]))

    out = [f"logged points        {len(rows)}",
           f"iteration range      {rows[0]['it']} to {last_it}",
           f"final dacc           {rows[-1]['dacc']:.3f}",
           f"dacc over last 20%   {tail_acc:.3f}",
           f"final D(real)        {rows[-1]['dr']:+.2f}",
           f"final D(fake)        {rows[-1]['df']:+.2f}"]

    c = first_sustained(rows, lambda r: r["dacc"] >= 0.95, window)
    s = first_sustained(rows, lambda r: r["dacc"] <= 0.55, window)
    out.append(f"collapse onset       "
               f"{f'iteration {c}' if c is not None else 'none detected'}")
    out.append(f"stall onset          "
               f"{f'iteration {s}' if s is not None else 'none detected'}")

    if tail_acc > 0.92:
        out.append("verdict              discriminator has won, generator is not learning")
    elif tail_acc < 0.60:
        out.append("verdict              discriminator cannot separate, generator gets no signal")
    else:
        out.append("verdict              adversarial balance looks healthy at the end")
    return out


# ------------------------------------------------------------------ figures
def fig_loss(rows, out: Path, dpi: int):
    it = [r["it"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(it, [r["g"] for r in rows], lw=1.2, label="generator loss")
    ax.plot(it, [r["d"] for r in rows], lw=1.2, label="discriminator loss")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("loss")
    ax.set_title("Generator and discriminator loss")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_loss.png", dpi=dpi)
    plt.close(fig)


def fig_dscores(rows, out: Path, dpi: int):
    it = [r["it"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(it, [r["dr"] for r in rows], lw=1.2, label="D(real)")
    ax.plot(it, [r["df"] for r in rows], lw=1.2, label="D(generated)")
    ax.axhline(0.0, color="grey", lw=1, ls="--")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean discriminator logit")
    ax.set_title("Discriminator output on real and generated images")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_dscores.png", dpi=dpi)
    plt.close(fig)


def fig_dacc(rows, out: Path, dpi: int):
    it = [r["it"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(it, [r["dacc"] for r in rows], lw=1.2, color="tab:red")
    ax.axhline(0.95, color="black", lw=1, ls=":", label="collapse threshold (0.95)")
    ax.axhline(0.55, color="tab:orange", lw=1, ls=":", label="stall threshold (0.55)")
    ax.axhline(0.5, color="grey", lw=1, ls="--", label="chance (0.50)")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("discriminator accuracy")
    ax.set_title("Discriminator accuracy against real and generated batches")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_dacc.png", dpi=dpi)
    plt.close(fig)


def fig_panel(rows, out: Path, dpi: int):
    it = [r["it"] for r in rows]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.5))

    ax = axs[0, 0]
    ax.plot(it, [r["g"] for r in rows], lw=1.1, label="generator")
    ax.plot(it, [r["d"] for r in rows], lw=1.1, label="discriminator")
    ax.set_title("(a) Adversarial loss")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3); ax.legend(frameon=False)

    ax = axs[0, 1]
    ax.plot(it, [r["dr"] for r in rows], lw=1.1, label="D(real)")
    ax.plot(it, [r["df"] for r in rows], lw=1.1, label="D(generated)")
    ax.axhline(0.0, color="grey", lw=1, ls="--")
    ax.set_title("(b) Discriminator logits")
    ax.set_xlabel("iteration"); ax.set_ylabel("mean logit")
    ax.grid(alpha=0.3); ax.legend(frameon=False)

    ax = axs[1, 0]
    ax.plot(it, [r["dacc"] for r in rows], lw=1.1, color="tab:red")
    ax.axhline(0.95, color="black", lw=1, ls=":")
    ax.axhline(0.55, color="tab:orange", lw=1, ls=":")
    ax.axhline(0.5, color="grey", lw=1, ls="--")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("(c) Discriminator accuracy")
    ax.set_xlabel("iteration"); ax.set_ylabel("accuracy")
    ax.grid(alpha=0.3)

    ax = axs[1, 1]
    ax.plot(it, [r["kimg"] for r in rows], lw=1.1, color="tab:green")
    ax.set_title("(d) Training budget")
    ax.set_xlabel("iteration"); ax.set_ylabel("kimg (thousands of real images)")
    ax.grid(alpha=0.3)

    fig.suptitle("Adversarial training dynamics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig_panel.png", dpi=dpi)
    plt.close(fig)


def fig_progression(run: Path, out: Path, dpi: int, max_panels=5):
    pngs = sorted(run.glob("samples_*.png"),
                  key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not pngs:
        print("no samples_*.png found, skipping progression figure")
        return
    if len(pngs) > max_panels:
        pick = np.linspace(0, len(pngs) - 1, max_panels).round().astype(int)
        pngs = [pngs[i] for i in sorted(set(pick))]

    fig, axs = plt.subplots(1, len(pngs), figsize=(3.4 * len(pngs), 3.9))
    for ax, p in zip(np.atleast_1d(axs), pngs):
        ax.imshow(plt.imread(p))
        ax.set_title(f"iteration {int(re.search(r'(\\d+)', p.stem).group(1)):,}",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle("Generated samples during training (EMA weights)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_progression.png", dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/dcgan128_sn", help="run directory")
    ap.add_argument("--log", default=None, help="defaults to <run>/log.jsonl")
    ap.add_argument("--out", default=None, help="defaults to the run directory")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()

    run = Path(a.run)
    log = Path(a.log) if a.log else run / "log.jsonl"
    out = Path(a.out) if a.out else run
    out.mkdir(parents=True, exist_ok=True)

    rows = load_log(log)
    print(f"reading {log}")
    for line in balance_report(rows):
        print("  " + line)

    fig_loss(rows, out, a.dpi)
    fig_dscores(rows, out, a.dpi)
    fig_dacc(rows, out, a.dpi)
    fig_panel(rows, out, a.dpi)
    fig_progression(run, out, a.dpi)

    for name in ("fig_loss", "fig_dscores", "fig_dacc", "fig_panel", "fig_progression"):
        p = out / f"{name}.png"
        if p.exists():
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
