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

Milestone 3 is where your text-to-image pipeline stops being a working prototype and starts being a measured system. At this checkpoint, your team computes quantitative metrics (FID and/or Inception Score) on a held-out set, conducts a qualitative prompt-to-image alignment analysis, runs a parameter-sensitivity study across at least three experimental dimensions, and produces a draft of the final report. Module 12 covers the multimodal landscape (CLIP, DALL-E 2, Stable Diffusion) in depth — you now have the conceptual vocabulary to interpret what your metrics are actually telling you about text–image alignment.

By the end of this milestone, your team should be able to answer two questions confidently: How good is our pipeline, numerically? and Where does it succeed and fail, and why? The draft report is the dress rehearsal for Milestone 4 — treat it as a real document, not a placeholder.

## What you will do

Working as a team, you will complete the following before the end of Module 12:

- Compute quantitative metrics. Compute FID (Fréchet Inception Distance) and/or Inception Score on a held-out set of generated images. Use a reproducible tool (e.g., clean-fid for FID, torchmetrics for IS). Report numerical values, the held-out sample sizes, and the reference distribution used. Briefly analyze what the numbers mean for your pipeline — high vs low scores, what they capture, what they miss.
- Conduct qualitative prompt-to-image alignment analysis. Curate a representative set of prompts (covering both success and failure cases) and analyze how well the generated images match the prompt semantics. Use CLIP-based alignment scoring or a structured human evaluation. Surface meaningful insights — which prompt categories work, which fail, and what patterns emerge in the failures.
- Conduct quantitative prompt-to-image alignment analysis. Curate a representative set of prompts (covering both success and failure cases) and analyze how well the generated images match the prompt semantics. Use CLIP-based alignment scoring or a structured human evaluation. Surface meaningful insights — which prompt categories work, which fail, and what patterns emerge in the failures.
- Run a parameter-sensitivity study. Vary at least three dimensions across: guidance scale, noise schedule, number of sampling steps, sampler choice (DDPM vs DDIM), and embedding choice (e.g., CLIP vs BERT, if applicable). Visualize the trade-offs (quality vs compute, alignment vs diversity) in clear plots or tables.
- Comparative evaluation. If you have access to two model variants (e.g., your fine-tuned model vs the baseline pretrained pipeline from Milestone 1, or two different conditioning configurations), compare them on the same prompts and same metrics. This produces the comparison your final report will hinge on.
- Draft the final report. Produce a working draft of the 6–8 page final report — introduction, background, methodology, experimental setup, results (tables and plots from steps 1–4), and a preliminary discussion section. The ethical-considerations section can be a stub at this point but should be outlined.

##Deliverables

Submit the following via Canvas:

- Link to your team's updated GitHub repository (commit hash or release tag at the time of submission)
- Evaluation notebook — reproducible computation of FID/IS, alignment scoring, and parameter-sensitivity sweeps, with all plots and tables generated inline
- Draft final report (PDF, target 4–6 pages at draft stage) — introduction, methodology, experimental setup, results, and preliminary discussion; ethics section may be outlined
- Generated samples used in evaluation — held-out images, prompts, and seeds organized in the repo so a reviewer can verify the metrics
- Plots and tables — FID/IS values, alignment scores, parameter-sensitivity visualizations, and any comparative evaluation results

One team member should submit on behalf of the team. Make sure the GitHub repo remains private and that all team members and the teaching staff have read access.

## Looking ahead

Milestone 4 (end of Module 13) is the final write-up. The draft report, evaluation tables, sensitivity plots, and generated samples you produce here are the raw material of the final report and gallery. Between Milestone 3 and Milestone 4 you will sharpen the discussion, expand the ethical-considerations section into a substantive ~1-page reflection on the dual-use nature of generative AI, curate the final gallery of prompt→image pairs (success and failure cases), and clean the repository for reproducibility.

The final presentation in Module 14 is graded on a separate rubric — you will prepare slides and demo material as part of Milestone 4, but the live presentation itself is evaluated separately.


# Milestone 4

Milestone 4 is the capstone deliverable of Project 2. At this checkpoint, your team submits the final report (technically detailed and ethically reflective), a curated gallery of generated prompt→image results spanning successes and failures, a complete, reproducible GitHub repository with thorough documentation, and the presentation slides + demo you will deliver in Module 14. This is where the dataset work from Milestone 1, the integrated pipeline from Milestone 2, and the evaluation work from Milestone 3 are pulled together into a single polished artifact.

The Module 13 timing is deliberate: Music Generation is the final generative-domain module of the course, and its themes (cross-modal generation, controllability, evaluation difficulty) echo the issues you are now writing about in your own report. Use the breathing room created by no new assignments in M13 to push the report and gallery from "draft" to "publishable."

## What you will do

Working as a team, you will complete the following before the end of Module 13:

- Write the final report (6–8 pages, PDF). Structure: introduction and motivation, background and related work, methodology and model architecture, experimental setup and results (tables and plots from Milestone 3), analysis and discussion (including parameter sensitivity and model comparisons), and an ethical-considerations and conclusion section (~1 page minimum). Engage seriously with the dual-use nature of generative AI — consent in training data, deepfakes, copyright, environmental cost, representation and bias. Acknowledgement alone is not sufficient.
- Curate the generated results gallery. Assemble at least 20 prompt→image examples that demonstrate the range of your pipeline's behavior: strong successes, partial successes, characteristic failures, and edge cases. Each example should be labeled with the prompt, key parameter settings (guidance scale, sampler, seed), and a short caption explaining what it illustrates.
- Finalize the GitHub repository. The repo must be reproducible: pinned dependencies, an environment specification, a top-level README.md with clear setup / training / inference / evaluation instructions, organized notebooks, and a structure a reviewer can navigate without help. Include the dataset card from Milestone 1, the integration write-up from Milestone 2, and the evaluation artifacts from Milestone 3.
- Prepare the presentation slides and demo (delivered live in Module 14). Build a 15–20 minute slide deck covering goals, approach, key findings, and reflection, ready for a live demonstration of your pipeline generating images from new prompts. Slides and demo are submitted as Milestone 4 deliverables; the live presentation in M14 is graded on the separate Final Project Presentation rubric.
- Repository hygiene pass. Confirm everything in the repo runs from a clean clone, the README reproduces the headline results, and all team contributions are visible in the commit history.

## Deliverables

Submit the following via Canvas:

- Final report (PDF, 6–8 pages) — full structure listed above, including the ethical-considerations section
- Generated results gallery (PDF or curated folder in the repo) — ≥20 prompt→image examples with labels, parameter settings, and captions
- Link to your team's final GitHub repository (commit hash or release tag pinned for grading) — reproducible, with a thorough README
- Presentation slides (PDF or link to the deck) — the 10–15 minute Module 14 presentation
- Demo material — a working notebook or demo assets a reviewer can run to reproduce the demo prompts, ready to drive your live Module 14 demonstration

One team member should submit on behalf of the team. After grading, the repo may be opened to public visibility at the team's discretion; until then keep it private with read access for the teaching staff.

## Looking ahead

Module 14 is the capstone: your team delivers the 15–20 minute presentation and live demonstration via Zoom at a scheduled slot, responds to questions from faculty and peers, and completes the peer assessment of other teams' work (each session is recorded and posted for review). The Module 14 presentation is graded on the separate Final Project Presentation rubric (technical content, demo quality, analysis and reflection, presentation quality, team participation) — review that rubric in advance so your team prepares to the criteria.

Once the presentation is complete, the course closes with cross-cohort reflection on the discriminative→generative arc you have traveled since Module 1. Submit a clean, well-documented repository here in Milestone 4: it is both your portfolio artifact and the basis on which a faculty reviewer or future employer will judge the engineering quality of the project.

# Final Project Presentation

The Final Project Presentation is the capstone of this course. Each team presents their Generative Vision Model Project (Project 2) live via Zoom at a scheduled slot, then engages with peers and faculty through the presentation discussion in Module 14 (each session is recorded and posted for review). The presentation is embedded within the Project 2 grade — there is no separate grade weight for Module 14. There is no new technical content to complete; this module is dedicated entirely to reflection, presentation, and peer review.
Presentation Logistics

- Format: A live presentation and demonstration via Zoom. Each team signs up for a presentation time slot and presents live; during the session the instructional team provides multiple prompts on the fly, and the team demonstrates the image-generation capabilities of its integrated model by producing images that accurately match the given prompts. Cover a brief overview of the system architecture, a live demonstration of the complete pipeline, and a discussion of key findings, model performance, and limitations. Approximately 15–20 minutes; every team member must make a substantive contribution to the prepared presentation content, and only one team member must be present to deliver it. If none of the team can attend the scheduled slot due to documented medical need, time difference, or other unforeseeable circumstances, contact the TAs in the beginning of the term (by the end of Week 3) to arrange an alternative time and/or record the presentation under the supervision of the TAs.
- Length: 10–15 minutes total. Every team member must narrate a portion of the presentation.
- Sharing & discussion: Recordings are posted to the Module 14 presentation discussion. During the Module 14 window, view your peers' presentations and post questions on their threads; each team responds to the questions on its own thread before the module closes.
- Slides: Prepare a slide deck and submit it (PDF or link) alongside your recording and your Project 2 notebook/code deliverable.
- Recording & peer review: Your live session is recorded and posted to the Module 14 presentation discussion so peers and faculty can review it and complete the peer assessment. Attendance at your scheduled slot is expected; see the format note above for the documented-exception process.

What Your Presentation Should Cover

- Project motivation: What problem did you set out to solve? Why is generative modeling the right approach?
- Architecture and design choices: Which generative model(s) did you implement and why? How do they connect to what you learned in Modules 6–13?
- Training procedure: Dataset, preprocessing, training setup, and any engineering challenges encountered.
- Results and evaluation: Quantitative metrics (FID, IS, LPIPS, or task-appropriate measures) and qualitative analysis of generated outputs.
- Responsible AI reflection: What ethical or societal implications does your system carry? Address at least one concern from the following: bias in generated outputs, dual-use potential, copyright attribution, or environmental cost of training.
- What you would do differently: Honest reflection on limitations and next steps if the project were to continue.

## Peer Review Component

Once recordings are posted, each student completes a brief peer review of another team's project within the Module 14 window. Instructions and the peer review form will be distributed separately. Peer Assessment is evaluated on thoughtfulness and specificity, not on how positively you rate your peers.

## What to Submit
- Your slide deck (PDF or PowerPoint), submitted to this assignment page within the Module 14 window; your live session recording is posted after you present (or upload a recording if you arranged the recorded alternative with the TAs).
- Your Project 2 notebook and any supplementary code should already be submitted on the Project 2 assignment page in Module 13.

