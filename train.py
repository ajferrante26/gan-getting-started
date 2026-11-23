"""train.py - unconditional DCGAN on the 300 Monet paintings.

Kaggle: gan-getting-started ("I'm Something of a Painter Myself").
Trains at 128x128. gen_submission.py upsamples to 256 for scoring.

Discriminator uses spectral normalization and the default loss is hinge, which
bounds how confident the discriminator can become. --loss bce restores the
original cross-entropy objective for comparison.

python train.py --iters 30000 --batch 32 --out runs/dcgan128_sn
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.parametrizations import spectral_norm
from torchvision import transforms
from torchvision.utils import save_image

IMG_EXTS = {".jpg", ".jpeg", ".png"}

DATA_CANDIDATES = [
    Path("/kaggle/input/gan-getting-started"),
    Path("/data"),
    Path("data"),
]


# ------------------------------------------------------------- path resolving
def _images_in(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def resolve_images(root: str | Path) -> tuple[Path, list[Path]]:
    """Return (directory, sorted image paths).

    Tries, in order: a monet_jpg (or monet) subfolder anywhere under root, root
    itself, then a single subfolder that contains images.
    """
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"--data does not exist: {root.resolve()}")

    for name in ("monet_jpg", "monet"):
        for d in [root / name] + [p for p in root.rglob(name) if p.is_dir()]:
            files = _images_in(d)
            if files:
                return d, files

    files = _images_in(root)
    if files:
        return root, files

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    hits = [(d, _images_in(d)) for d in subdirs]
    hits = [(d, f) for d, f in hits if f]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        names = ", ".join(d.name for d, _ in hits)
        raise ValueError(
            f"{root} has several image folders ({names}). "
            f"Point --data at the one you want, e.g. --data {hits[0][0]}"
        )
    raise FileNotFoundError(
        f"no {'/'.join(sorted(IMG_EXTS))} files found under {root.resolve()}."
    )


def resolve_default() -> Path:
    for cand in DATA_CANDIDATES:
        try:
            resolve_images(cand)
            return cand
        except (FileNotFoundError, ValueError):
            continue
    raise FileNotFoundError("no dataset found. Pass --data.")


def load_images(root: str | Path, size: int) -> tuple[Path, list[Path], torch.Tensor]:
    """Decode every image once into a [N,3,size,size] float tensor."""
    d, paths = resolve_images(root)
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    xs = [tf(Image.open(p).convert("RGB")) for p in paths]
    return d, paths, torch.stack(xs)


# ------------------------------------------------------------- networks
def weights_init(m):
    """SN-aware: with a parametrized weight, initialize the raw tensor."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if "weight" in getattr(m, "parametrizations", {}):
            nn.init.normal_(m.parametrizations.weight.original, 0.0, 0.02)
        else:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, z_dim=128, base=32, img=128, ch=3):
        super().__init__()
        self.z_dim = z_dim
        self.init_size = img // 16
        self.c0 = base * 16
        self.fc = nn.Linear(z_dim, self.c0 * self.init_size ** 2)
        blocks, c = [], self.c0
        while c > base:
            blocks += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        self.blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(nn.Conv2d(c, ch, 3, 1, 1), nn.Tanh())

    def forward(self, z):
        h = self.fc(z).view(z.size(0), self.c0, self.init_size, self.init_size)
        return self.to_rgb(self.blocks(h))


class Discriminator(nn.Module):
    """Every convolution is spectrally normalized. No batch normalization."""

    def __init__(self, base=32, img=128, ch=3, max_ch=256):
        super().__init__()
        n_down = int(math.log2(img)) - 2
        chans = [ch] + [min(base * 2 ** i, max_ch) for i in range(n_down)]
        blocks = []
        for i in range(n_down):
            blocks.append(spectral_norm(
                nn.Conv2d(chans[i], chans[i + 1], 4, 2, 1, bias=True)))
            blocks.append(nn.LeakyReLU(0.2, True))
        blocks.append(spectral_norm(nn.Conv2d(chans[-1], 1, 4, 1, 0)))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).flatten(1)


# ------------------------------------------------------- augmentation, EMA, R1
def diff_augment(x, policy="color,translation,cutout"):
    """Simplified DiffAugment (Zhao et al., 2020), fully vectorized.

    Branch decisions use Python's random module, so nothing here touches the
    device and no CUDA sync is forced.
    """
    n = x.size(0)
    if "color" in policy:
        if random.random() < 0.5:
            x = x + (torch.rand(n, 3, 1, 1, device=x.device) - 0.5) * 0.2
        if random.random() < 0.5:
            g = torch.rand(n, 1, 1, 1, device=x.device) * 0.8 + 0.6
            x = x * g + (torch.rand(n, 3, 1, 1, device=x.device) - 0.5) * 0.1
    if "translation" in policy:
        shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * 0.125
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, :, 2] = shift
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
    if "cutout" in policy:
        h, w = x.shape[-2:]
        cut = int(0.35 * min(h, w))
        cy = torch.randint(0, h, (n, 1, 1), device=x.device)
        cx = torch.randint(0, w, (n, 1, 1), device=x.device)
        yy = torch.arange(h, device=x.device).view(1, h, 1)
        xx = torch.arange(w, device=x.device).view(1, 1, w)
        keep = ((yy - cy).abs() > cut // 2) | ((xx - cx).abs() > cut // 2)
        x = x * keep.unsqueeze(1)
    return x.clamp(-1, 1)


class EMA:
    def __init__(self, module, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(module).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, m):
        for e, s in zip(self.module.state_dict().values(), m.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(s, alpha=1 - self.decay)
            else:
                e.copy_(s)


def r1_penalty(d_out, x_real, gamma):
    grad = torch.autograd.grad(d_out.sum(), x_real, create_graph=True)[0]
    return gamma * 0.5 * grad.pow(2).flatten(1).sum(1).mean()


# ---------------------------------------------------------------- plots
def make_plots(log_path: Path, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "it" in r]
    if not rows:
        return
    it = [r["it"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(it, [r["g"] for r in rows], label="generator loss")
    ax.plot(it, [r["d"] for r in rows], label="discriminator loss")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title("Adversarial loss")
    fig.tight_layout(); fig.savefig(out_dir / "fig_loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(it, [r["dr"] for r in rows], label="D(real)")
    ax.plot(it, [r["df"] for r in rows], label="D(fake)")
    ax.axhline(0.0, color="grey", lw=1, ls="--")
    ax.set_xlabel("iteration"); ax.set_ylabel("mean logit")
    ax.legend(); ax.set_title("Discriminator output on real vs generated")
    fig.tight_layout(); fig.savefig(out_dir / "fig_dscores.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="runs/dcgan128")
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--z", type=int, default=128)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--g-lr", type=float, default=2e-4)
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["hinge", "bce"], default="hinge")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--r1", type=float, default=0.0,
                    help="0 off. redundant with spectral norm, kept for comparison")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast. do not combine with spectral norm")
    ap.add_argument("--policy", default="color,translation,cutout")
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device} torch={torch.__version__}")

    d, paths, data = load_images(a.data if a.data else resolve_default(), a.img)
    print(f"images: {len(paths)} from {d}")
    print(f"first file: {paths[0].name}")
    data = data.to(device)
    n_img = data.size(0)

    use_amp = a.amp and device == "cuda"
    if use_amp and a.r1 > 0:
        print("R1 needs fp32 gradients, disabling AMP for this run")
        use_amp = False
    if use_amp:
        print("note: AMP with spectral normalization is untested here")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"amp={use_amp} loss={a.loss} r1={a.r1} policy={a.policy} batch={a.batch}")

    G = Generator(a.z, a.base, a.img).to(device).apply(weights_init)
    D = Discriminator(a.base, a.img).to(device).apply(weights_init)
    ema = EMA(G, a.ema) if a.ema > 0 else None

    opt_g = torch.optim.Adam(G.parameters(), lr=a.g_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=a.d_lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, a.z, device=device)
    log = open(out / "log.jsonl", "w")
    run = {"d": 0.0, "g": 0.0, "dr": 0.0, "df": 0.0, "acc": 0.0, "n": 0}
    t0 = time.time()

    for step in range(1, a.iters + 1):
        idx = torch.randint(0, n_img, (a.batch,), device=device)
        real = data[idx]
        flip = torch.rand(a.batch, 1, 1, 1, device=device) < 0.5
        real = torch.where(flip, real.flip(-1), real)

        # ---- discriminator
        z = torch.randn(a.batch, a.z, device=device)
        with torch.no_grad():
            fake = G(z)
        with torch.amp.autocast("cuda", enabled=use_amp):
            d_real = D(diff_augment(real, a.policy))
            d_fake = D(diff_augment(fake, a.policy))
            if a.loss == "hinge":
                loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            else:
                loss_d = bce(d_real, torch.ones_like(d_real)) + \
                         bce(d_fake, torch.zeros_like(d_fake))
        if a.r1 > 0:
            real_r1 = real.detach().clone().requires_grad_(True)
            loss_d = loss_d + r1_penalty(D(real_r1), real_r1, a.r1)
        opt_d.zero_grad(set_to_none=True)
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)

        # ---- generator
        z = torch.randn(a.batch, a.z, device=device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            g_out = D(diff_augment(G(z), a.policy))
            loss_g = (-g_out.mean() if a.loss == "hinge"
                      else bce(g_out, torch.ones_like(g_out)))
        opt_g.zero_grad(set_to_none=True)
        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()

        if ema is not None:
            ema.update(G)

        with torch.no_grad():
            run["d"] += loss_d.item(); run["g"] += loss_g.item()
            run["dr"] += d_real.float().mean().item()
            run["df"] += d_fake.float().mean().item()
            run["acc"] += ((d_real > 0).float().mean()
                           + (d_fake < 0).float().mean()).item() / 2
            run["n"] += 1

        if step % 100 == 0:
            n = run["n"]
            rec = {"it": step, "kimg": step * a.batch / 1000,
                   "d": run["d"] / n, "g": run["g"] / n,
                   "dr": run["dr"] / n, "df": run["df"] / n, "dacc": run["acc"] / n}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"it {step:6d} kimg {rec['kimg']:7.1f} "
                  f"d {rec['d']:.3f} g {rec['g']:.3f} "
                  f"D(real) {rec['dr']:+.2f} D(fake) {rec['df']:+.2f} "
                  f"dacc {rec['dacc']:.2f} {time.time()-t0:6.0f}s")
            run = {k: 0.0 for k in run}

        if step % a.sample_every == 0:
            src = ema.module if ema is not None else G
            src.eval()
            with torch.no_grad():
                imgs = src(fixed_z)
            src.train()
            save_image(imgs, out / f"samples_{step:06d}.png",
                       nrow=8, normalize=True, value_range=(-1, 1))
            torch.save({"g": G.state_dict(),
                        "g_ema": (ema.module if ema else G).state_dict(),
                        "args": vars(a)}, out / f"ckpt_{step:06d}.pt")

    log.close()
    torch.save({"g": G.state_dict(),
                "g_ema": (ema.module if ema else G).state_dict(),
                "args": vars(a)}, out / "ckpt.pt")
    json.dump(vars(a), open(out / "args.json", "w"), indent=2)
    make_plots(out / "log.jsonl", out)
    print(f"done. artifacts in {out}")


if __name__ == "__main__":
    main()
