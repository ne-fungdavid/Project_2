Notes. 

### 300-epoch result

**Bottom row of `train_progress_epoch300(1).png` (row 6) is epoch 300.** That's the actual result you use for the report.

### Honest assessment

**The monkey at 300 epochs is genuinely good.** Two monkeys hugging, green background, clear structure. This is a massive leap from the 50-epoch version (multi-headed mess). It proves the model is learning *semantics*, not just patterns.

**Strong findings for the report:**
- **Cosine schedule beats linear** (beta-sweep): linear produces scrambled noise, cosine produces clear images. This is a concrete, defensible finding.
- **CFG works beautifully** (conditioning check): conditioned (monkey) vs unconditional (noise) shows the text encoder is actually doing something useful.
- **DDIM converges at 25-50 steps** (steps sweep): no quality difference between 25, 50, 200 steps. DDIM is efficient and effective.
- **Minimal overfitting** (loss curve): train and val are within ~0.01 of each other. The model generalizes well to the 500-image validation set.
- **Self-attention at 128×128 works** (75M vs 34M params): the monkey at 300 epochs is far more coherent than anything from 64×64 at 500 epochs.

**Clear limitations:**
- **DDPM fails** (1000 steps = all grey). The model hasn't learned to denoise at t>900. With 300 more epochs this might improve, but at 300 epochs it's a hard limit.
- **Prompt 1 (interior) is clean but bland.** White pillars, blue tint. No specific content.
- **Prompt 2 (dog) is simple but solid.** Sleeping dog, warm tones, but not highly detailed.
- **4,500 images is the bottleneck.** The model can recognize "monkey" and "dog" but can't generate novel poses or complex compositions.

### Verdict

The 300-epoch 128×128 result is **strong enough to stand on**. The monkey image alone is a convincing demonstration that text-conditional diffusion can work from scratch with limited data. The DDPM failure and the beta-sweep comparison give you concrete "why it works / why it fails" talking points.

**Next:** 256×256 at 30 epochs (smoke test, batch 32, LR 1e-4, same BASE_CHANNELS=96). This gives you the resolution sweep data point for the report. Then write up.