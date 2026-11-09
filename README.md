# Project 2 — Text-to-Image Generation

**IE 7615** | Northeastern University
**David Fung**

A text-to-image generative model combining a CLIP text encoder with a conditional diffusion model.

## Milestones

| # | Focus | Due | Status |
|---|---|---|---|
| 1 | Dataset, CLIP embeddings, baseline SD generation | End of Module 7 | Complete |
| 2 | Model integration, guidance/noise experiments | End of Module 10 | Complete |
| 3 | FID/IS evaluation, parameter sweeps | End of Module 12 | In progress |
| 4 | Final report, gallery, presentation | End of Module 13 | Planned |

## Milestone 1 (Complete)

`Milestone_1/DavidFung_Project2.Milestone1.ipynb` is a single, self-contained notebook with three tasks:

1. **Task 1 — Dataset curation**: loads Flickr30k (`umaru97/flickr30k_train_val_test`) from Hugging Face, cleans captions, filters by length (8–200 words), deduplicates, caps at 5,000 pairs, resizes to 512×512, and writes `data/curated/` + `manifest.csv`.
2. **Task 2 — CLIP embedding pipeline**: loads CLIP ViT-B/32, tokenizes a 75-pair sample, computes text & image embeddings (512-dim), reports cosine alignment, and projects the joint space with PCA.
3. **Task 3 — Baseline generation**: generates 16 prompt→image samples from Stable Diffusion v1-5 (512×512, guidance 10.0, 50 steps, seed 42 + index), compares each to its ground-truth Flickr image, and saves a side-by-side gallery.

Supporting documents: `Milestone_1/dataset_memo.md` (dataset memo) and `Milestone_1/proposal.md` (project proposal).

## Milestone 2 (Complete)

Trains a from-scratch text-conditional DDPM (U-Net + cosine schedule + classifier-free guidance, 64×64, CLIP ViT-L/14 text features) on the curated dataset, then runs the full prompt → embedding → conditioning → image pathway.

- `Milestone_2/Integration_Training.ipynb` — end-to-end training (best val 0.0417 over 1000 epochs; loss curve + training-progress GIF in `outputs/` / `gallery/`).
- `Milestone_2/Conditional_Generation.ipynb` — 16-prompt baseline gallery with ground-truth comparisons, plus guidance / steps / schedule / sampler sweeps.

## Milestone 3 (In progress)

`Milestone_3/Evaluation.ipynb` evaluates the Milestone 2 model: FID + Inception Score on a held-out reference set, CLIP prompt-to-image alignment (quantitative + qualitative), a parameter-sensitivity study consolidating the Milestone 2 sweeps, and a comparison against SD v1.5 on the same prompts and metrics.

## Milestone 4 (Planned)

`Milestone_4/Demo.ipynb` is a working prompt-to-image demo (any prompt, a few seconds per image).
The 20+ example labeled gallery (with SD v1.5 comparison) is built by a small
`build_gallery.py` script that records prompt / seed / parameters per example.

## Structure

```
Project_2/
├── README.md
├── .gitignore
├── data/
│   └── curated/                    # 5000 image + caption pairs (M1)
├── Milestone_1/                    # M1 notebook + baseline SD gallery
├── Milestone_2/                    # M2 notebooks, outputs, gallery, memos
├── Milestone_3/                    # Evaluation.ipynb — FID/IS, alignment, sensitivity, SD comparison
├── Milestone_4/                    # Demo.ipynb — live prompt-to-image demo + labeled gallery
```

## Environment

- Python 3.10+ (verified on 3.14)
- PyTorch (CPU or CUDA — Milestone 1 completed on CPU)
- `transformers` (CLIP), `diffusers` (Stable Diffusion)
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pillow`, `tqdm`
- M2 training + generation ran on an **NVIDIA GB10** GPU (bf16, batch 256); M3–M4 use the same GPU
- `clean-fid` (FID) and `torchmetrics` (Inception Score) are required for Milestone 3

## Reproduce (Milestone 1)

```bash
pip install torch transformers diffusers pillow numpy pandas matplotlib scikit-learn tqdm
jupyter notebook Milestone_1/DavidFung_Project2.Milestone1.ipynb
```

- The dataset and both models download automatically from Hugging Face on first run.
- Runs on CPU or GPU. On CPU the 16-image baseline generation takes ~22 minutes; on a GPU it is much faster.
- Outputs are written to `data/curated/` and `Milestone_1/gallery/`.

## Reproduce (Milestones 2–4)

Requires the GB10 GPU environment (checkpoint, embeddings and gallery are already in `Milestone_2/outputs/` and `Milestone_2/gallery/`).

```bash
jupyter notebook Milestone_2/Integration_Training.ipynb   # or Conditional_Generation.ipynb
jupyter notebook Milestone_3/Evaluation.ipynb
jupyter notebook Milestone_4/Demo.ipynb
```
