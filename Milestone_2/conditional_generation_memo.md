# Milestone 2 — Conditional Generation & Parameter Sweeps (Memo)

**IE 7615 — Project 2**  |  David Fung  |  2026-11-01

Device: `cuda`  |  Model: `TextConditionedUNet` (34.3M params)  |  Resolution: 64×64  |  Schedule: `cosine`


---

## 1. Setup

- Trained model: `/home/davidfung/jupyterlab/project_2/Milestone_2/outputs/ddpm_best_val_l14_1000ep.pt` (best val loss **0.0417**).
- Text encoder: CLIP `openai/clip-vit-large-patch14` (frozen, 512-dim) — same as Milestone 1.
- Baseline generation: **16** curated prompts, guidance=**10**, 50 DDIM steps, seed 42+i, cosine schedule.
- Sweeps: guidance [1.0, 3.0, 7.5, 10.0, 15.0]; steps [25, 50, 100]; schedule ['linear', 'cosine']; sampler ['ddim', 'ddpm'].

### Baseline prompts (evenly spaced across the curated set)

  - **0** (id=0): two young guys with shaggy hair look at their hands while hanging out in the yard.
  - **1** (id=367): a man wearing a yellow shirt and tan pants plays golf outside while being watched by other people.
  - **2** (id=729): three people wearing colorful costumes and masks sitting between two stone statues.
  - **3** (id=1085): a man in jeans and a white t-shirt chops up vegetables in a kitchen.
  - **4** (id=1452): a man with a white shirt and rubber boots is climbing on tanks in order to retrieve items from a specific tank.
  - **5** (id=1807): the girl is wearing a white shirt with pink skirt, and she is standing in front of a house.
  - **6** (id=2166): a young man, in a baseball cap and blue striped shirt, is playing the guitar.
  - **7** (id=2524): a woman in a red hat sits on a winding purple and bright green bench with a bush in the background.
  - **8** (id=2873): a boy in blue and red goggles is swimming.
  - **9** (id=3231): a girl in a purple dress smiling as she climbs an indoor rock wall.
  - **10** (id=3586): the little boy with the yellow stripe on his hat is in the snow.
  - **11** (id=3933): a black dog is attempting to catch a thrown object that is red.
  - **12** (id=4288): a crowded stadium is full of people watching an event.
  - **13** (id=4639): a fluffy dog plays with a small toy in a living room.
  - **14** (id=4993): a man in a yellow reflective vest squats on a lawn while using on a cellphone.
  - **15** (id=5349): a man is standing inside a building under construction filled with bricks and various tools.


## 2. Baseline gallery

Ground-truth (top) vs generated (bottom): `gallery/baseline/baseline_comparison.png`. Individual images: `gallery/baseline/base_XX.png`.

Baseline timing: **0.48** s/image (mean).


## 3. Guidance-scale sweep

| Guidance `s` | Mean CLIP text–image align | Diversity | s/image |
|---|---|---|---|
| 1 | 0.1361 | 0.5396 | 0.47 |
| 3 | 0.1327 | 0.6147 | 0.47 |
| 7.5 | 0.1641 | 0.6182 | 0.47 |
| 10 | 0.1732 | 0.6006 | 0.47 |
| 15 | 0.1731 | 0.6245 | 0.47 |

Grid: `gallery/guidance/guidance_sweep.png`.

**Observations** *(complete after reviewing the grid):*

- Low `s`: __ (e.g., weaker/blurry prompt alignment, more diverse) __
- High `s`: __ (e.g., sharper/saturated, lower diversity) __
- Best trade-off observed at `s` = __ (justify using the table above) __

## 4. Noise schedule & sampling

**Steps (DDIM):**

| Steps | s/image |
|---|---|
| 25 | 0.23473604520161948 |
| 50 | 0.4642847379048665 |
| 100 | 0.930733839670817 |

**Schedule (DDIM, 50 steps):**

| Schedule | s/image |
|---|---|
| linear | 0.4680956204732259 |
| cosine | 0.46430420875549316 |

**Sampler (guidance 10):**

| Sampler | s/image |
|---|---|
| DDIM | 0.46840206782023114 |
| DDPM | 9.30046558380127 |

Grids: `steps_sweep.png`, `schedule_sweep.png`, `sampler_sweep.png`.

**Observations** *(complete after reviewing the grids):*

- Steps: __ (fidelity vs. speed) __
- Schedule: __ (linear vs cosine difference) __
- Sampler: __ (DDIM vs DDPM — DDPM runs all T=1000 steps; ~20x slower) __

## 5. Conditioning check

To confirm the model is *conditional* (not just image-plausible), compare a prompt's output with the same seed using the unconditional (null) embedding (guidance≈0). They should differ and the guided one should match the text. Record the qualitative difference here.


## 6. Reproducibility

- Seeds: baseline 42+i; sweeps 42+i per prompt.
- Full parameter manifest: `outputs/generation_manifest.json` (52 generations, each with prompt, seed, guidance, steps, schedule, sampler, time, output path).
- Rerun: `Milestone_2/Integration_Training.ipynb` then `Milestone_2/Conditional_Generation.ipynb`.
