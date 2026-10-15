# Project 2

Project 2 asks your team to design, implement, evaluate, and present a text-to-image generative model that combines a pretrained text encoder (CLIP or BERT) with a conditional diffusion model. Project 2 is the major summative assessment of the course and synthesizes the generative modeling techniques developed across Modules 6–13.

- Milestone 1 — Project Setup and Dataset Preparation (end of Module 7): Image–caption dataset prepared; GPU environment and baseline text→embedding pipeline functional; baseline prompt→image runs; initial project proposal with team roles.
- Milestone 2 — Model Integration and Baseline Results (end of Module 10): CLIP or BERT integrated with the diffusion model; baseline conditional image generation working; guidance scale and noise schedule experimentation documented.
- Milestone 3 — Evaluation and Analysis (end of Module 12): FID and Inception Score computed and analyzed; qualitative prompt-to-image alignment evaluated; parameter sensitivity sweeps; draft final report.
- Milestone 4 — Final Report and Presentation (end of Module 13): Complete final report with ethical reflection; generated-results gallery; full GitHub repo; presentation prepared for Module 14.
- Final Presentation (Module 14): a live team presentation and demonstration via Zoom at a scheduled slot, recorded and posted for peer and faculty review; graded against its own rubric (see the Module 14 Final Project Presentation page).

## Milestone 1

### Overview

Milestone 1 is the first graded checkpoint for Project 2: the text-to-image generative pipeline built on a Transformer text encoder and a diffusion-based image generator. At this checkpoint, your team assembles the foundational pieces of the pipeline into a single, graded deliverable: a curated image–caption dataset, a working GPU environment with a text→embedding pipeline, an initial project proposal with team roles, and a baseline gallery of prompt→image samples generated from a pretrained Stable Diffusion (or equivalent) pipeline.

With the discriminative project (Project 1) complete in Module 5, Milestone 1 marks the moment Project 2 becomes your team's primary focus through Module 13. By now you have also studied the Transformer (Module 7), which directly informs the text-encoder choices you will make in the next milestone.

### What you will do

Working as a team, you will complete and submit the following as a single Milestone 1 package:

1. Select and preprocess your image–caption dataset. Choose a dataset (COCO Captions, Flickr30k, Conceptual Captions, or a curated LAION subset) and curate a working subset of roughly 1k–5k image–caption pairs. Apply consistent image resizing, normalization, and a caption-cleaning pipeline (length filters, deduplication, quality heuristics, optional content filter). Document license, domain coverage, and ethical considerations in a short dataset memo.
2. Set up your environment and text→embedding pipeline. Install PyTorch + Hugging Face Diffusers + Hugging Face Transformers + CLIP in a reproducible environment (Colab, RCC GPU, or local). Build a notebook that loads CLIP, tokenizes captions, computes image and text embeddings for a sample of pairs, and verifies the pipeline runs end-to-end without errors.
3. Run baseline prompt→image generation. Generate images for 10–20 representative prompts drawn from your curated dataset using a pretrained Stable Diffusion (or equivalent) pipeline. Compare generated outputs against ground-truth images in your dataset and write a short qualitative memo using generative-modeling vocabulary (latent representation, conditional sampling, mode collapse risk, generative vs discriminative framing).
4. Write the initial project proposal. A 1–2 page proposal that clearly explains your dataset choice, the model and conditioning approach you plan to use through Milestones 2 and 3, the team roles for the generative work, and the risks you anticipate (compute budget, dataset licensing, evaluation feasibility).
5. Stand up the team GitHub repository. Initialize a repo that will house all Project 2 code. Include the dataset-preparation notebook, the CLIP embedding notebook, the baseline Stable Diffusion generation notebook, the dataset memo, the proposal, and a README that documents how to reproduce your environment and rerun the baseline.

### Deliverables

Submit the following via Canvas:

- Link to your team's GitHub repository (commit hash or release tag at the time of submission)
- Initial project proposal (PDF, 1–2 pages) — dataset, model plan, team roles, anticipated risks
- Dataset memo (PDF, 1–2 pages) — dataset choice, size, license, domain coverage, ethical considerations
- Baseline generation gallery and memo (PDF or Markdown export, 2–3 pages) — the 10–20 prompt→image samples with side-by-side ground-truth comparisons and the qualitative analysis
- Notebook artifacts in the repo — at minimum a dataset-preparation notebook, a CLIP/text-embedding notebook, and a baseline Stable Diffusion generation notebook

### Looking ahead

Milestone 1 is the foundation on which the remaining three milestones are built. In Milestone 2 (end of Module 10), you will integrate a CLIP or BERT text encoder with a diffusion model, run conditional generation on baseline prompts, and experiment with guidance scale and noise schedule. The diffusion theory behind that work is taught in Module 9, giving you time to prepare before the milestone is due.

In Milestone 3 (end of Module 12) you will compute FID and Inception Score, run a parameter-sensitivity study, and draft the final report. Milestone 4 (end of Module 13) packages the final report, generated-results gallery, and complete repository; the final presentation follows in Module 14. Every later milestone builds directly on the dataset, environment, and repo you stand up here — invest the time now to make them clean and reproducible.

# Milestone 2

Milestone 2 is the first deep-work checkpoint of Project 2. You now combine the two halves of the text-to-image architecture you have been studying: a Transformer-based text encoder (CLIP or BERT) and a diffusion-based image generator. By the end of this milestone, your team will have an integrated pipeline that takes a text prompt as input, encodes it into a conditioning signal, and produces a generated image — with documented training runs, sampled outputs, and observations from experimenting with guidance scale and noise schedule.

This milestone is timed at the end of Module 10. The diffusion theory you need to make sense of denoising steps, noise schedules, and classifier-free guidance is taught in Module 9; this placement gives you through Module 10 to assemble, run, and document a working baseline without the crunch of submitting in the same week the theory is introduced. Use the dataset, environment, and embedding pipeline you stood up in Milestone 1 as your starting point — do not rebuild them.

### What you will do

Working as a team, you will complete the following before the end of Module 10:

1. Integrate the text encoder with a diffusion model. Wire CLIP or BERT embeddings into a diffusion pipeline as the conditioning signal. You may train an unconditional DDPM from scratch on a small subset and extend it to text-conditional, or you may build on a pretrained conditional pipeline (e.g., Stable Diffusion) and fine-tune or modify its conditioning path. Document the integration architecture clearly so a reviewer can trace text→embedding→conditioning→image flow.
2. Run baseline conditional generation. Generate images for a fixed set of baseline prompts drawn from your curated dataset. Outputs should be relevant to the prompt and demonstrate that the conditioning pathway is functional — not just that the diffusion model can produce any image. Save generated samples, the prompts that produced them, and the random seeds.
3. Experiment with guidance scale. Sweep classifier-free guidance scale (or equivalent conditioning strength) across at least three settings on the same prompts. Record how output quality, prompt alignment, and diversity change. Pick representative outputs to show in your write-up.
4. Experiment with noise schedule and sampling. Vary at least one of: noise schedule (linear, cosine, etc.), number of sampling steps, or sampler choice (DDPM vs DDIM). Compare quality vs compute trade-offs and document what you observe.
5. Capture training logs and observations. Save loss curves (or sampler diagnostics for inference-only pipelines), training/inference time, GPU memory, and any instability you encountered. Write a 1–2 page summary of what worked, what did not, and what you would change next.

If your team is targeting the from-scratch DDPM track, target a small image resolution (32×32 or 64×64) so training fits within your compute budget. Document this choice and its rationale in your summary.

### Deliverables

Submit the following via Canvas:

- Link to your team's updated GitHub repository (commit hash or release tag at the time of submission)
- Integration write-up (PDF, 1–2 pages) — architecture diagram, how text embeddings flow into the diffusion model, and any modifications to a pretrained pipeline
- Generated samples gallery (PDF or notebook export) — baseline conditional outputs plus the guidance-scale and noise-schedule comparison images, with prompts and parameter settings labeled
- Training logs and summary (PDF or Markdown export, 1–2 pages) — loss curves, runtime metrics, parameter-tuning observations, and lessons learned
- Notebook artifacts in the repo — at minimum an integration/training notebook and a conditional-generation notebook with parameter sweeps

### Looking ahead

The integrated pipeline you submit here is what you will evaluate in Milestone 3 and write up in Milestone 4. In Milestone 3 (end of Module 12), you will compute FID and Inception Score, conduct prompt-to-image alignment analysis, and run a parameter-sensitivity study across at least three dimensions (guidance scale, noise schedule, sampling steps, embedding choice). The Multimodal Models content in M12 gives you the conceptual grounding to interpret CLIP-based alignment scoring and the broader text–image embedding landscape.

Milestone 4 (end of Module 13) turns everything into the final report, generated-results gallery, and complete repository. Keep your training logs, seeds, and parameter settings organized now — you will need them for the parameter-sensitivity study and the report tables.

# Milestone 3

# Milestone 4