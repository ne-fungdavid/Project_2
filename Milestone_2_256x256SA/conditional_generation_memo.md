# Milestone 2 — Conditional Generation & Parameter Sweeps (Memo)

**IE 7615 — Project 2**  |  David Fung  |  2026-10-30

Device: `cuda`  |  Model: `TextConditionedUNet` (75.4M params)  |  Resolution: 256×256  |  Schedule: `cosine`


---

## 1. Setup

- Trained model: `outputs/ddpm_text_conditional.pt` (best val loss **0.0270**).
- Text encoder: CLIP `openai/clip-vit-base-patch32` (frozen, 512-dim) — same as Milestone 1.
- Baseline generation: **4** curated prompts, guidance=**10**, 50 DDIM steps, seed 42+i, cosine schedule.
- Sweeps: guidance [1.0, 3.0, 7.5, 10.0, 15.0]; steps [25, 50, 100]; schedule ['linear', 'cosine']; sampler ['ddim', 'ddpm'].

### Baseline prompts (evenly spaced across the curated set)

  - **0** (id=0): two young guys with shaggy hair look at their hands while hanging out in the yard.
  - **1** (id=1807): the girl is wearing a white shirt with pink skirt, and she is standing in front of a house.
  - **2** (id=3586): the little boy with the yellow stripe on his hat is in the snow.
  - **3** (id=5349): a man is standing inside a building under construction filled with bricks and various tools.


## 2. Baseline gallery

Ground-truth (top) vs generated (bottom): `gallery/baseline/baseline_comparison.png`. Individual images: `gallery/baseline/base_XX.png`.

Baseline timing: **7.94** s/image (mean).


## 3. Guidance-scale sweep

| Guidance `s` | Mean CLIP text–image align | Diversity | s/image |
|---|---|---|---|
| 1 | 0.224 | 0.6886 | 7.93 |
| 3 | 0.2237 | 0.6899 | 7.94 |
| 7.5 | 0.2187 | 0.6699 | 7.95 |
| 10 | 0.2136 | 0.6421 | 7.95 |
| 15 | 0.2221 | 0.6473 | 7.95 |

Grid: `gallery/guidance/guidance_sweep.png`.

**Observations** *(complete after reviewing the grid):*

- Low `s`: __ (e.g., weaker/blurry prompt alignment, more diverse) __
- High `s`: __ (e.g., sharper/saturated, lower diversity) __
- Best trade-off observed at `s` = __ (justify using the table above) __

## 4. Noise schedule & sampling

**Steps (DDIM):**

| Steps | s/image |
|---|---|
| 25 | 3.9380857149759927 |
| 50 | 7.871858437856038 |
| 100 | 15.761686563491821 |

**Schedule (DDIM, 50 steps):**

| Schedule | s/image |
|---|---|
| linear | 7.879529317220052 |
| cosine | 7.880313873291016 |

**Sampler (guidance 10):**

| Sampler | s/image |
|---|---|
| DDIM | 7.881551583607991 |
| DDPM | 158.24241065979004 |

Grids: `steps_sweep.png`, `schedule_sweep.png`, `sampler_sweep.png`.

**Observations** *(complete after reviewing the grids):*

- Steps: __ (fidelity vs. speed) __
- Schedule: __ (linear vs cosine difference) __
- Sampler: __ (DDIM vs DDPM — DDPM runs all T=1000 steps; ~20x slower) __

## 5. Conditioning check

To confirm the model is *conditional* (not just image-plausible), compare a prompt's output with the same seed using the unconditional (null) embedding (guidance≈0). They should differ and the guided one should match the text. Record the qualitative difference here.


## 6. Reproducibility

- Seeds: baseline 42+i; sweeps 42+i per prompt.
- Full parameter manifest: `outputs/generation_manifest.json` (40 generations, each with prompt, seed, guidance, steps, schedule, sampler, time, output path).
- Rerun: `Milestone_2/Integration_Training.ipynb` then `Milestone_2/Conditional_Generation.ipynb`.
