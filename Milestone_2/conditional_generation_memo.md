# Milestone 2 — Conditional Generation & Parameter Sweeps (Memo)

**IE 7615 — Project 2**  |  David Fung  |  2026-11-09

Device: `cuda`  |  Model: `TextConditionedUNet` (34.3M params)  |  Resolution: 64×64  |  Schedule: `cosine`


---

## 1. Setup

- Trained model: `/home/davidfung/jupyterlab/project_2/Milestone_2/outputs/ddpm_text_conditional_l14_1000ep.pt` — the **final epoch-1000 checkpoint** is used for all generation below (best val loss **0.0417**, recorded at epoch 268; see §9 checkpoint-selection note).
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
- Mid `s`: alignment peaks at `s`=3 (0.1858) and `s`=7.5 (0.1827), then *declines* by `s`=10–15 (~0.168) — extra guidance over-regularizes rather than improving fidelity.
- Diversity: peaks at `s`=10 (0.7266), then collapses at `s`=15 (0.566) as guidance suppresses sampling variability.
- Best trade-off: `s` = **3** — highest CLIP text–image alignment (0.1858) with diversity (0.6305) close to the `s`=1 level; `s`=10 is the choice if maximal diversity is preferred.
- Note: with the final checkpoint the alignment curve plateaus by `s`≈3, whereas the earlier best-val checkpoint kept rising through `s`=15 — the guidance optimum is checkpoint-dependent.

## 4. Noise schedule & sampling

**Steps (DDIM):**

| Steps | s/image |
|---|---|
| 25 | 0.23342402776082358 |
| 50 | 0.4660516579945882 |
| 100 | 0.930091142654419 |

**Schedule (DDIM, 50 steps):**

| Schedule | s/image |
|---|---|
| linear | 0.46692220369974774 |
| cosine | 0.4663965702056885 |

**Sampler (guidance 10):**

| Sampler | s/image |
|---|---|
| DDIM | 0.4663276672363281 |
| DDPM | 9.303540706634521 |

Grids: `steps_sweep.png`, `schedule_sweep.png`, `sampler_sweep.png`.

**Observations**

- Steps: per-image time scales ~linearly with step count (25 → 0.24 s, 50 → 0.46 s, 100 → 0.93 s). Probes at 32/100/200: 32 ≈ 50 (no visible difference), 100 marginally better, 200 slightly *worse* at s=10 (guidance is re-applied at every step, so it over-regularizes over long trajectories). DDIM quality therefore plateaus by ~50 steps — the sampler is not the binding constraint; residual quality is limited by data scale (~4.5k pairs), the 64×64 resolution, and model capacity (34.3M params, global conditioning).
- Schedule: linear and cosine are statistically tied (0.467 vs 0.466 s; visually comparable) — cosine kept for the baseline.
- Sampler: DDIM (0.47 s) is ~20× faster than DDPM (9.30 s) because DDIM runs 50 steps while DDPM must run all T=1000 steps.

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


## 7. Resolution study — 128×128 & 256×256 (sensitivity runs)

Two sensitivity-analysis (SA) runs tested the same architecture at higher resolution (75.4M params, `BASE_CHANNELS=96`, batch 32, B/32 512-dim CLIP), archived in `Milestone_2_128x128SA/` and `Milestone_2_256x256SA/`:

| Run | CLIP | U-Net | Batch | Epochs | Wall time | s/epoch | Best val | Prompts | s/image (DDIM-50) |
|---|---|---|---|---|---|---|---|---|---|
| **64×64 (primary)** | L/14 (768-d) | 34.3M | 256 | 1000 | 8.2 h | 29.7 | 0.0417 | 16 | 0.47 |
| 128×128 (SA) | B/32 (512-d) | 75.4M | 32 | 50 (→300 follow-up) | 3.5 h (50 ep) | 252 (~8.5×) | 0.0352 (0.0295 @300 ep) | 4 | 1.83 |
| 256×256 (SA) | B/32 (512-d) | 75.4M | 32 | 60 | 16.1 h | 965 (~32×) | 0.0261 | 4 | 7.82 |

**128×128.** Per-epoch cost is ~8.5× the 64×64 run. The 300-epoch follow-up showed genuine semantic learning (e.g., two monkeys hugging — a clear step up from the multi-headed 50-epoch outputs) and reproduced the cosine-beats-linear and CFG findings, but: (a) the DDPM mode was non-functional (all-grey at T=1000); (b) compositions were limited by the 4.5k training set; (c) only 4 prompts were evaluated.


**256×256.** Per-epoch cost is ~32×; 60 epochs (16.1 h) is still undertrained for that resolution, only 4 prompts were evaluated, and the run showed no quality advantage over the 64×64 primary within the available budget.


These SA runs serve as the scale study for the resolution choice — they quantify the cost curve (≈resolution² per step × undertrained budgets) rather than competing with the primary result.


## 8. Why 64×64 is the preferred direction

1. **Budget vs. data coverage.** 64×64 ran 1,000 epochs (4.5M image steps) in 8.2 h, whereas 128×128 saw 50 epochs in 3.5 h (~8.5× per-epoch cost) and 256×256 saw 60 epochs in 16.1 h (~32× per-epoch cost; 2× the wall time for ~1/16 of the epochs). The 64×64 run saw the dataset ~17–20× more often.

2. **Quality per cost.** The 64×64 final checkpoint is the best prompt-fidelity result in the project (verified side-by-side, including against its own best-val checkpoint and the 128/256 SA outputs); the undertrained 128/256 runs (≤300 epochs, batch 32) did not surpass it, and 128's DDPM mode failed outright.

3. **Data is the binding constraint.** The same 4.5k pairs are relatively scarcer at 128×128/256×256: the 128 run shows subject recognition without compositional diversity, i.e., the ceiling is data scale — not the sampler or the resolution.

4. **Scope.** 64×64 fully exercises the Milestone 2 objectives (CLIP integration, CFG + null token, AdaGN conditioning, all four sweeps, the conditioning check, CLIP metrics); resolution is not the milestone objective, and the 128/256 runs serve exactly the scale-study role.

5. **Encoder.** 64×64 is the only run using the upgraded CLIP L/14 (768-d) text features; 128/256 used B/32 (512-d) — a further 64×64 advantage (confound acknowledged).

6. **The right next step for resolution** is the pretrained SD v1.5 comparison in the next milestone: the SA runs quantify that from-scratch 256 needs orders of magnitude more compute/data, while 64×64 plus pretrained transfer delivers the resolution gain within the project budget.


## 9. Notes

**Checkpoint selection.** All generation in this memo uses the *final* epoch-1000 checkpoint rather than the best-val checkpoint (best val loss 0.0417). Side-by-side generation showed the final checkpoint matches prompts noticeably better (subjects, colors, scenes) despite its higher validation loss — for this small U-Net + CLIP-L/14 setup, validation loss is a poor proxy for generation quality.


**Resolution scope.** Covered in §7–§8: 128×128/256×256 SA runs are archived and superseded by the 64×64 L/14 primary result.
