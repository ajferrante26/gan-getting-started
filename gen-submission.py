"""gen_submission.py - 10,000 generated images at 256x256, zipped for Kaggle.

Loads the EMA generator weights from a train.py checkpoint, samples fresh noise,
upsamples to 256, and writes a flat zip with no directory structure.

    python gen_submission.py --ckpt runs/dcgan128_sn/ckpt_030000.pt --out images.zip
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import torch
from PIL import Image

from train import Generator


def load_ckpt(path: Path) -> dict:
    """torch.load defaults to weights_only=True from torch 2.6 onward."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"weights_only load failed ({exc.__class__.__name__}), retrying unrestricted")
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/dcgan128/ckpt.pt")
    ap.add_argument("--n", type=int, default=10000, help="must be 7,000 to 10,000")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fmt", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--preview", default=None,
                    help="optional path for a 64-image preview grid")
    ap.add_argument("--out", default="images.zip")
    a = ap.parse_args()

    if not 7000 <= a.n <= 10000:
        raise SystemExit(f"--n must be between 7,000 and 10,000, got {a.n}")

    torch.manual_seed(a.seed)
    ck = load_ckpt(Path(a.ckpt))
    cfg = ck["args"]
    G = Generator(cfg["z"], cfg["base"], cfg["img"])
    G.load_state_dict(ck["g_ema"])
    G.eval()
    print(f"ckpt      {a.ckpt}")
    print(f"generator img={cfg['img']} z={cfg['z']} base={cfg['base']} "
          f"policy={cfg.get('policy')} r1={cfg.get('r1')} loss={cfg.get('loss')}")
    print(f"output    {a.n} images at {a.size}x{a.size}, format={a.fmt}")

    preview = []
    written = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as zf:
        for start in range(0, a.n, a.batch):
            b = min(a.batch, a.n - start)
            with torch.no_grad():
                x = G(torch.randn(b, cfg["z"]))
            x = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()

            for arr in x:
                im = Image.fromarray(arr).resize((a.size, a.size), Image.LANCZOS)
                buf = io.BytesIO()
                if a.fmt == "jpg":
                    im.save(buf, format="JPEG", quality=a.quality)
                else:
                    im.save(buf, format="PNG")
                zf.writestr(f"{written}.{a.fmt}", buf.getvalue())
                written += 1
                if len(preview) < 64:
                    preview.append(im.copy())

            if written % 1000 == 0:
                print(f"  {written}/{a.n}")

    with zipfile.ZipFile(a.out) as zf:
        names = zf.namelist()
    assert len(names) == a.n, f"zip holds {len(names)} entries, expected {a.n}"
    assert not any("/" in n or "\\" in n for n in names), "zip has directory structure"
    assert all(n.lower().endswith((".jpg", ".jpeg", ".png")) for n in names)

    size_mb = Path(a.out).stat().st_size / 1e6
    print(f"wrote {written} images to {a.out} ({size_mb:.0f} MB)")
    print(f"first entries: {names[:3]}")

    if a.preview:
        cols, rows = 8, 8
        w, h = preview[0].size
        sheet = Image.new("RGB", (cols * w, rows * h))
        for i, im in enumerate(preview[:cols * rows]):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sheet.save(a.preview)
        print(f"preview grid -> {a.preview}")


if __name__ == "__main__":
    main()
