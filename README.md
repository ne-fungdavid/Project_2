# Project 2 — Text-to-Image Generation

**IE 7615** | Northeastern University
**David Fung**

A text-to-image generative model combining a CLIP text encoder with a conditional diffusion model.

## Milestones

| # | Focus | Due | Status |
|---|---|---|---|
| 1 | Dataset, CLIP embeddings, baseline SD generation | End of Module 7 | Complete |
| 2 | Model integration, guidance/noise experiments | End of Module 10 | — |
| 3 | FID/IS evaluation, parameter sweeps | End of Module 12 | — |
| 4 | Final report, gallery, presentation | End of Module 13 | — |

## Milestone 1 (this submission)

`Milestone_1/DavidFung_Project2.Milestone1.ipynb` is a single, self-contained notebook with three tasks:

1. **Task 1 — Dataset curation**: loads Flickr30k (`umaru97/flickr30k_train_val_test`) from Hugging Face, cleans captions, filters by length (8–200 words), deduplicates, caps at 5,000 pairs, resizes to 512×512, and writes `data/curated/` + `manifest.csv`.
2. **Task 2 — CLIP embedding pipeline**: loads CLIP ViT-B/32, tokenizes a 75-pair sample, computes text & image embeddings (512-dim), reports cosine alignment, and projects the joint space with PCA.
3. **Task 3 — Baseline generation**: generates 16 prompt→image samples from Stable Diffusion v1-5 (512×512, guidance 10.0, 50 steps, seed 42 + index), compares each to its ground-truth Flickr image, and saves a side-by-side gallery.

Supporting documents: `Milestone_1/dataset_memo.md` (dataset memo) and `Milestone_1/proposal.md` (project proposal).

## Structure

```
Project_2/
├── README.md
├── .gitignore
├── Milestone_1/
│   ├── DavidFung_Project2.Milestone1.ipynb
│   ├── pca.png                     # CLIP joint-space PCA plot
│   └── gallery/                    # baseline + comparison images
├── Milestone_2/
├── Milestone_3/
├── Milestone_4/
```

## Environment

- Python 3.10+ (verified on 3.14)
- PyTorch (CPU or CUDA — Milestone 1 completed on CPU)
- `transformers` (CLIP), `diffusers` (Stable Diffusion)
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pillow`, `tqdm`

## Reproduce (Milestone 1)

```bash
pip install torch transformers diffusers pillow numpy pandas matplotlib scikit-learn tqdm
jupyter notebook Milestone_1/DavidFung_Project2.Milestone1.ipynb
```

- The dataset and both models download automatically from Hugging Face on first run.
- Runs on CPU or GPU. On CPU the 16-image baseline generation takes ~22 minutes; on a GPU it is much faster.
- Outputs are written to `data/curated/` and `Milestone_1/gallery/`.
