# Milestone 2 — Conditional Generation & Parameter Sweeps (Memo)

**IE 7615 — Project 2**  |  David Fung  |  2026-11-08

Device: `cuda`  |  Model: `TextConditionedUNet` (34.3M params)  |  Resolution: 64×64  |  Schedule: `cosine`


---

## 1. Setup

- Trained model: `/home/davidfung/jupyterlab/project_2/Milestone_2/outputs/ddpm_text_conditional_l14_1000ep.pt` — the **final epoch-1000 checkpoint** is used for all generation below (best val loss **0.0417**, recorded at epoch 268; see §7 checkpoint-selection note).
- Text encoder: CLIP `openai/clip-vit-large-patch14` (frozen, 768-dim; upgraded from Milestone 1's B/32).
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

Baseline timing: **0.47** s/image (mean).


## 3. Guidance-scale sweep

| Guidance `s` | Mean CLIP text–image align | Diversity | s/image |
|---|---|---|---|
| 1 | 0.1258 | 0.6117 | 0.47 |
| 3 | 0.1858 | 0.6305 | 0.47 |
| 7.5 | 0.1827 | 0.6666 | 0.46 |
| 10 | 0.168 | 0.7266 | 0.47 |
| 15 | 0.1682 | 0.566 | 0.46 |

Grid: `gallery/guidance/guidance_sweep.png`.

**Observations**

- Low `s`: at `s`=1 prompt alignment is weakest (0.1258) — outputs stay diverse (0.6117) but drift away from the text.
- Mid `s`: alignment peaks at `s`=3 (0.1858) and `s`=7.5 (0.1827), then *declines* to ~0.168 by `s`=10–15 — extra guidance over-regularizes rather than improving fidelity.
- Diversity: peaks at `s`=10 (0.7266), then collapses at `s`=15 (0.566) as guidance suppresses sampling variability.
- Best trade-off: `s` = **3** — highest CLIP text–image alignment (0.1858) with diversity (0.6305) close to the `s`=1 level; `s`=10 is the choice if maximal diversity is preferred.
- Note: with the final checkpoint the alignment curve plateaus by `s`≈3, whereas the earlier best-val checkpoint kept rising through `s`=15 — the guidance optimum is checkpoint-dependent.

## 4. Noise schedule & sampling

**Steps (DDIM):**

| Steps | s/image |
|---|---|
| 25 | 0.23551313082377115 |
| 50 | 0.4646290938059489 |
| 100 | 0.9315536816914877 |

**Schedule (DDIM, 50 steps):**

| Schedule | s/image |
|---|---|
| linear | 0.46650854746500653 |
| cosine | 0.466950257619222 |

**Sampler (guidance 10):**

| Sampler | s/image |
|---|---|
| DDIM | 0.46677470207214355 |
| DDPM | 9.330320358276367 |

Grids: `steps_sweep.png`, `schedule_sweep.png`, `sampler_sweep.png`.

**Observations**

- Steps: per-image time scales ~linearly with step count (25 → 0.24 s, 50 → 0.46 s, 100 → 0.93 s); 50 steps is the sweet spot — ~2× faster than 100 with near-identical quality.
- Schedule: linear and cosine are statistically tied (0.467 vs 0.467 s; visually comparable) — cosine kept for the baseline.
- Sampler: DDIM (0.47 s) is ~20× faster than DDPM (9.33 s) because DDIM runs 50 steps while DDPM must run all T=1000 steps.

## 5. Conditioning check

To confirm the model is *conditional* (not just image-plausible), compare a prompt's output with the same seed using the unconditional (null) embedding (guidance≈0). They should differ and the guided one should match the text.

**Result** (prompt "two young guys with shaggy hair look at their hands while hanging out in the yard", seed 42, `conditioning_check.png`):

- `s`=7.5 (conditioned): two figures in a green yard, heads down looking at their hands — matches the prompt's subjects, pose and setting.
- `s`=0 (same seed, null embedding): a lone person in a red shirt by a body of water — a completely different scene with no trace of the prompt.
- Conclusion: with the seed held fixed and only the conditioning changed, the output tracks the text embedding → the model is genuinely *conditional*, not merely image-plausible.


## 6. Reproducibility

- Seeds: baseline 42+i; sweeps 42+i per prompt.
- Full parameter manifest: `outputs/generation_manifest.json` (52 generations, each with prompt, seed, guidance, steps, schedule, sampler, time, output path).
- Rerun: `Milestone_2/Integration_Training.ipynb` then `Milestone_2/Conditional_Generation.ipynb`.

---

## 7. Notes

**Resolution (reduced scope).** 128×128 and 256×256 sensitivity runs (B/32, 100 epochs; best val 0.0295 / 0.0270) are archived in `Milestone_2_128x128SA/` and `Milestone_2_256x256SA/`. 64×64 with CLIP L/14 is kept as the primary resolution — it leaves the most compute headroom and, with the final checkpoint, delivers the best prompt fidelity in this project.

**Checkpoint selection.** All generation in this memo uses the *final* epoch-1000 checkpoint rather than the best-val checkpoint (epoch 268, val 0.0417). Side-by-side generation showed the final checkpoint matches prompts noticeably better (subjects, colors, scenes) despite its higher validation loss — for this small U-Net + CLIP-L/14 setup, validation loss is a poor proxy for generation quality.
