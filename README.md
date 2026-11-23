# Monet-Style Image Synthesis with a DCGAN

Unconditional deep convolutional GAN trained from scratch on the 300 Monet
paintings from Kaggle's **GANs Getting Started** competition (*I'm Something of a
Painter Myself*). Built for CSC-8150 Assignment 7.

The project trains three configurations that each fail differently, which is the
main result. A plain DCGAN collapses into discriminator dominance, adding an R1
gradient penalty reduces the severity without preventing the outcome, and spectral
normalization with hinge loss overshoots into a discriminator that cannot separate
real from generated images at all. The memorization analysis comes out clean.

**Note on the assignment brief.** The Problem and Purpose bullet asks for "realistic
dog images." That line does not match the dataset or the metric. The competition is
Monet paintings scored by MiFID, and this project addresses that task.

---

## Dataset

| Property | Value |
|---|---|
| Competition | `gan-getting-started` |
| Training images | 300 Monet paintings, 256x256 JPEG in `monet_jpg/` |
| Unused | 7,028 photographs in `photo_jpg/`, relevant only to CycleGAN-style translation |
| Also present | `monet_tfrec/`, `photo_tfrec/`, ignore unless using TF input pipelines |
| Metric | MiFID, memorization-informed Fréchet Inception Distance, lower is better |
| Submission | 7,000 to 10,000 images at 256x256, flat zip, no directory prefix |

Expected layout:

```
assignment7/
  data/
    monet_jpg/        <- 300 files, this is what gets trained on
    photo_jpg/
  train.py
  gen_submission.py
  plots.py
  get_numbers.py
  runs/
```

`train.py` resolves `--data` against a `monet_jpg` subfolder, a flat folder of
JPEGs, or a competition root, so passing `data` or `data/monet_jpg` both work.

---

## Files

| File | Purpose |
|---|---|
| `train.py` | Loads images, builds the GAN, trains, writes checkpoints, samples, and logs |
| `gen_submission.py` | Samples 10,000 images from a checkpoint, upsamples to 256, writes and validates `images.zip` |
| `plots.py` | Figures and balance report from `log.jsonl`. No torch import |
| `get_numbers.py` | Every paper number that does not require Kaggle, including the memorization check |
| `runs/<name>/` | Checkpoints, config, log, sample grids, figures |

---

## Requirements

Python 3.10 or newer. A CUDA-capable GPU is strongly recommended.

```
torch>=2.0
torchvision
pillow
matplotlib
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib pillow
```

On Windows, `pip install torch` from plain PyPI resolves to a **CPU-only** wheel.
Check before training:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A version string ending in `+cpu` with `None` and `False` means the wheel is wrong.

---

## Quickstart

```powershell
# 1. train (about 46 minutes at 128px on an RTX 5000 class card)
python train.py --iters 30000 --batch 32 --out runs/dcgan128_sn

# 2. figures and the balance report
python plots.py --run runs/dcgan128_sn

# 3. 10,000 images at 256x256, validated
python gen_submission.py --ckpt runs/dcgan128_sn/ckpt_030000.pt `
                         --out images.zip `
                         --preview runs/dcgan128_sn/fig_submitted.png

# 4. every number the write-up needs
python get_numbers.py --runs runs/dcgan128 runs/dcgan128_sn `
                      --memorization runs/dcgan128_sn/ckpt_030000.pt
```

Upload `images.zip` to the competition, then record the MiFID score and the
submission timestamp.

---

## train.py

```
python train.py [options]
```

| Option | Default | Notes |
|---|---|---|
| `--data` | auto | Falls back to the Kaggle mount, then `/data`, then `./data` |
| `--out` | `runs/dcgan128` | Checkpoints, log, samples, figures |
| `--img` | `128` | Training resolution. Submission is upsampled to 256 |
| `--z` | `128` | Noise dimension |
| `--base` | `32` | Base channel width. Generator reaches `base*16` channels at 8x8 |
| `--batch` | `32` | 300 images means a `batch` of 32 is roughly one tenth of the dataset |
| `--iters` | `30000` | Iteration based, not epoch based |
| `--g-lr` | `2e-4` | Adam, betas 0.5 and 0.999 |
| `--d-lr` | `1e-4` | Deliberately half the generator rate, which is a suspect setting |
| `--loss` | `hinge` | `hinge` or `bce` |
| `--ema` | `0.999` | Generator weight averaging, used for all samples and submissions |
| `--r1` | `0.0` | R1 gradient penalty coefficient. Redundant with spectral norm |
| `--amp` | off | fp16 autocast. Do not combine with spectral norm |
| `--policy` | `color,translation,cutout` | DiffAugment operations applied to discriminator inputs |
| `--sample-every` | `1000` | Sample grid and numbered checkpoint interval |
| `--seed` | `42` | Split, initialization, and noise |

### Architecture

Generator: 128-dim noise, linear projection to 512x8x8, four transposed convolutions
(512 to 256 to 128 to 64 to 32) with batch normalization and ReLU, then a 3x3
convolution to three channels with tanh. Roughly 7 million parameters.

Discriminator: five strided convolutions from 3 to 32 to 64 to 128 to 256 to 256
channels, each spectrally normalized, leaky ReLU at 0.2, then a 4x4 convolution to a
scalar logit. No batch normalization. Roughly 1.7 million parameters.

### Augmentation

DiffAugment applies colour jitter, sub-pixel translation, and rectangular cutout to
discriminator inputs on both real and generated batches. Every operation is
vectorized and every branch decision comes from Python's `random` module, so the
augmentation forces no CUDA synchronization. Branch decisions made with
`torch.rand(()) < 0.5` cost about 200 device synchronizations per iteration and were
removed for that reason.

### Data handling

All 300 images are decoded once into a single `[300, 3, 128, 128]` tensor held on
the GPU. Batches are drawn by random index on the device, and horizontal flips are
applied with `torch.where`. There is no `DataLoader`. With 300 images and a batch of
32, an epoch is 9 iterations, so a `DataLoader` with workers respawned its processes
every 9 iterations on Windows and cost more time than the forward and backward
passes combined. Removing it and vectorizing the augmentation took throughput from
0.84 to 0.084 seconds per iteration at identical settings.

### Adversarial losses

Hinge: the discriminator minimizes `relu(1 - D(real)) + relu(1 + D(fake))` and the
generator minimizes `-mean(D(fake))`. At `D = 0` on every input the discriminator
loss equals 2.0, which is a useful diagnostic.

BCE: two binary cross-entropy terms over logits with non-saturating generator loss.

---

## Interpreting the training log

`dacc` is the mean of the fraction of real batches scored positive and the fraction
of generated batches scored negative. It is the single most diagnostic number in the
output and it is more useful than either loss.

| `dacc` | Meaning | Action |
|---|---|---|
| 0.60 to 0.85 | Healthy adversarial balance | Let it run |
| Above 0.95 for 2,000+ iterations | Discriminator has won, generator receives no gradient | Reduce `--d-lr`, add regularization |
| Below 0.55 for 2,000+ iterations | Discriminator cannot separate, generator receives no signal | Raise `--d-lr`, check the loss function |

Both failure modes appeared in this project and produced near-identical generator
loss curves, so check `dacc` before drawing conclusions from loss values.

---

## gen_submission.py

```
python gen_submission.py [options]
```

| Option | Default | Notes |
|---|---|---|
| `--ckpt` | `runs/dcgan128/ckpt.pt` | **Always pass this explicitly** |
| `--n` | `10000` | Must be between 7,000 and 10,000, validated before generation |
| `--size` | `256` | Submission resolution |
| `--batch` | `64` | Sampling batch size |
| `--fmt` | `jpg` | `jpg` or `png` |
| `--quality` | `95` | JPEG quality |
| `--seed` | `1234` | Noise seed, separate from the training seed |
| `--preview` | none | Writes an 8x8 sheet of the first 64 submitted images |
| `--out` | `images.zip` | Output archive |

Loads `g_ema`, not the raw generator weights. Verifies entry count, flat structure,
and image extensions by reopening the archive. A successful run ends with
`first entries: ['0.jpg', '1.jpg', '2.jpg']`.

---

## plots.py

```
python plots.py --run runs/dcgan128_sn
```

Reads `log.jsonl` and writes `fig_loss.png`, `fig_dscores.png`, `fig_dacc.png`,
`fig_panel.png`, and `fig_progression.png`. Also prints a balance report with the
collapse onset, the stall onset, and a verdict.

---

## get_numbers.py

```
python get_numbers.py --runs runs/dcgan128 runs/dcgan128_sn `
                      --memorization runs/dcgan128_sn/ckpt_030000.pt
```

Prints final and last-20-percent values for each run, both failure onsets, and the
four memorization distances in pixel space. Distances are raw L2 over 49,152 pixel
values and are also reported as RMS per channel so they can be read against the
0 to 2 scale of the image data.

---

## Outputs

| File | Written by | Contents |
|---|---|---|
| `log.jsonl` | `train.py` | One record per 100 iterations: losses, logits, `dacc`, `kimg` |
| `args.json` | `train.py` | Full configuration for the run |
| `ckpt_NNNNNN.pt` | `train.py` | Generator and EMA weights plus args every `--sample-every` |
| `ckpt.pt` | `train.py` | Final checkpoint, written once at the end |
| `samples_NNNNNN.png` | `train.py` | 8x8 grid of 64 fixed-noise samples |
| `fig_loss.png`, `fig_dscores.png` | both | Adversarial loss and discriminator logits |
| `fig_dacc.png`, `fig_panel.png`, `fig_progression.png` | `plots.py` | Accuracy, four-panel summary, sample progression |
| `images.zip` | `gen_submission.py` | 10,000 flat JPEG entries |
| `fig_submitted.png` | `gen_submission.py` | What MiFID actually scored |

---

## Recorded runs

Hardware was a Quadro RTX 5000 with 16 GB under PyTorch 2.11.0+cu128, float32,
seed 42, batch 32, 128x128 training resolution.

| Run | Objective | Iterations | kimg | Wall clock | s/iter | Outcome |
|---|---|---|---|---|---|---|
| `dcgan128` | BCE, DiffAugment, EMA | 30,000 | 960 | 2,437 s | 0.081 | Collapsed at iteration 14,300 |
| `dcgan128_r1` | plus R1 at 5 | 40,000 | 1,280 | 5,179 s | 0.129 | Collapsed, less severely |
| `dcgan128_sn` | hinge plus spectral norm | 30,000 | 960 | 2,775 s | 0.093 | Stalled at iteration 2,500 |

R1 costs 1.59x per iteration because the penalty differentiates through the
discriminator's own gradient. Spectral normalization costs 1.14x because of the
power iteration on every convolution.

`dcgan128` final: `dacc` 0.999, last-20-percent mean 0.988, `D(real)` +5.374,
`D(fake)` -4.997, discriminator loss 0.032, generator loss 5.564.

`dcgan128_r1` final 2,000 iterations: `dacc` 0.985, `D(real)` +3.88, `D(fake)` -3.70.
The 40,000-iteration log for this run was destroyed by a relaunch, see Known Issues.

`dcgan128_sn` final: `dacc` 0.517, last-20-percent mean 0.511, `D(real)` -0.031,
`D(fake)` -0.064, discriminator loss 1.971, generator loss 0.159. A hinge
discriminator loss of 2.0 with both logits near zero means the discriminator scores
every input identically.

### Memorization check

`dcgan128_sn/ckpt_030000.pt`, 1,000 generated images against 300 training images.

| Statistic | Value | RMS per channel |
|---|---|---|
| Generated to train, mean | 74.68 | 0.3369 |
| Generated to train, minimum | 46.57 | 0.2101 |
| Train to train, mean | 77.82 | 0.3510 |
| Train to train, minimum | 32.05 | 0.1446 |

Minimum ratio 1.45. The closest generated image is farther from any training
painting than the two most similar training paintings are from each other, so no
replication is detected. The generated side had 300,000 candidate pairs against
89,700 for the training side, which makes the conclusion conservative.

### MiFID

Pending. Fill in after the first scored upload, together with the submission ID and
timestamp.

---

## Known Issues

**`log.jsonl` is opened with mode `w`, so relaunching into an existing `--out`
directory destroys the previous history.** It also overwrites `ckpt_*.pt` of the
same iteration numbers. This cost the 40,000-iteration log of `dcgan128_r1`, leaving
only a 2,500-iteration log from an unintended configuration in its place. Before
adding a safety check, either use a distinct `--out` per run or archive the directory
first. A guard along these lines prevents a repeat:

```python
if (out / "log.jsonl").exists() and not a.overwrite:
    raise SystemExit(f"{out} already holds a run, pass --overwrite or choose a new --out")
```

**`gen_submission.py` defaults to `runs/dcgan128/ckpt.pt`, which is the collapsed
first run.** Always pass `--ckpt`.

**`make_plots` imports matplotlib inside the function**, so a missing dependency
crashes after training completes rather than at startup. Install matplotlib before
the first run.

**`--amp` is untested with spectral normalization.** The power iteration is
sensitive to reduced precision. Leave `--amp` off.

**The discriminator learning rate is half the generator's.** Both published recipes
this project draws from use equal rates. The asymmetry is a plausible contributor to
run 3's stall.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False on Windows | PyPI served a CPU-only wheel | Reinstall from `download.pytorch.org/whl/cu128` after `pip uninstall torch torchvision` |
| `ModuleNotFoundError: No module named 'matplotlib'` after a long run | Dependency missing | `pip install matplotlib pillow` |
| `FileNotFoundError: no .jpg files found` | Wrong `--data` | Pass the folder containing the JPEGs, or the competition root |
| Only 7,028 images loaded | Resolver picked `photo_jpg` | Pass `--data data/monet_jpg` explicitly |
| `dacc` reaches 0.99 and stays | Discriminator dominance | Lower `--d-lr`, check regularization |
| `dacc` sits at 0.50 with logits near zero | Discriminator cannot separate | Raise `--d-lr` toward `--g-lr`, try `--loss bce` |
| Training far slower than expected | Host-side data pipeline | Confirm the in-GPU tensor path is in use, expect about 0.08 s/iter at 128px |
| Zip rejected by the scorer | Directory structure or wrong count | The validation step in `gen_submission.py` catches both |

---

## Reproducibility

Every run records its own configuration in `args.json` and its per-batch EMA
checkpoints in `ckpt_NNNNNN.pt`. To regenerate a result, use the recorded `seed`,
`img`, `z`, `base`, `loss`, `r1`, `policy`, `ema`, and learning rates, plus the
library versions printed at the top of the training log.

Two caveats. The generator uses its own submission seed of 1234, distinct from the
training seed of 42, and both belong in any write-up. And `dcgan128_r1` cannot be
exactly regenerated from disk because its log was truncated, so its reported numbers
exist only in saved terminal output.

Report in the paper: architecture widths, `img`, `z`, `base`, batch size, learning
rates, loss function, regularization, iteration count, kimg, wall clock, seed, GPU
model, torch and CUDA versions, and the MiFID score with its submission timestamp.

