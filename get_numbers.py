"""get_numbers.py - every paper number that does not require Kaggle.

    python get_numbers.py --runs runs/dcgan128 runs/dcgan128_sn
    python get_numbers.py --memorization runs/dcgan128_sn/ckpt_030000.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "it" in r and "dacc" in r:
            rows.append(r)
    rows.sort(key=lambda r: r["it"])
    return rows


def first_sustained(rows, pred, window):
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


def run_stats(path, window=2000):
    rows = load_rows(path)
    last = rows[-1]["it"]
    tail = [r for r in rows if r["it"] >= 0.8 * last]
    name = Path(path).parent.name
    print(f"--- {name} " + "-" * max(1, 44 - len(name)))
    print(f"  iterations          {rows[0]['it']} to {last}")
    print(f"  final dacc          {rows[-1]['dacc']:.3f}")
    print(f"  last20 dacc mean    {sum(r['dacc'] for r in tail)/len(tail):.3f}")
    print(f"  last20 dr mean      {sum(r['dr'] for r in tail)/len(tail):+.3f}")
    print(f"  last20 df mean      {sum(r['df'] for r in tail)/len(tail):+.3f}")
    print(f"  final dr / df       {rows[-1]['dr']:+.3f} / {rows[-1]['df']:+.3f}")
    print(f"  final d / g loss    {rows[-1]['d']:.3f} / {rows[-1]['g']:.3f}")
    c = first_sustained(rows, lambda r: r["dacc"] >= 0.95, window)
    s = first_sustained(rows, lambda r: r["dacc"] <= 0.55, window)
    print(f"  collapse onset      {c if c is not None else 'none'}")
    print(f"  stall onset         {s if s is not None else 'none'}")
    print()


def nn_distances(query, ref, exclude_self=False, bs=64):
    out = []
    for i in range(0, query.size(0), bs):
        q = query[i:i + bs].flatten(1)
        d = torch.cdist(q, ref.flatten(1))
        if exclude_self:
            idx = torch.arange(i, min(i + bs, query.size(0)))
            d[idx - i, idx] = float("inf")
        out.append(d.min(1).values)
    return torch.cat(out)


def memorize(ckpt, n=1000, seed=1234):
    from train import Generator, load_images, resolve_default

    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["args"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(cfg["z"], cfg["base"], cfg["img"]).to(dev)
    G.load_state_dict(ck["g_ema"])
    G.eval()

    root = cfg.get("data") or resolve_default()
    d, paths, train = load_images(root, cfg["img"])
    train = train.to(dev)

    torch.manual_seed(seed)
    with torch.no_grad():
        fake = torch.cat([G(torch.randn(128, cfg["z"], device=dev))
                          for _ in range((n + 127) // 128)])[:n]

    gen = nn_distances(fake.cpu(), train.cpu())
    trn = nn_distances(train.cpu(), train.cpu(), exclude_self=True)
    dim = 3 * cfg["img"] * cfg["img"]
    print(f"--- memorization, {n} generated vs {len(paths)} training at {cfg['img']}px")
    print(f"  [A] generated -> train  mean {gen.mean():7.2f}  rms {gen.mean()/dim**0.5:.4f}")
    print(f"  [B] generated -> train  min  {gen.min():7.2f}  rms {gen.min()/dim**0.5:.4f}")
    print(f"  [C] train     -> train  mean {trn.mean():7.2f}  rms {trn.mean()/dim**0.5:.4f}")
    print(f"  [D] train     -> train  min  {trn.min():7.2f}  rms {trn.min()/dim**0.5:.4f}")
    print(f"  min ratio gen/train     {gen.min()/trn.min():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--memorization", default=None)
    a = ap.parse_args()
    for r in a.runs:
        p = Path(r) / "log.jsonl"
        if p.exists():
            run_stats(p)
        else:
            print(f"missing {p}")
    if a.memorization:
        memorize(a.memorization)


if __name__ == "__main__":
    main()
