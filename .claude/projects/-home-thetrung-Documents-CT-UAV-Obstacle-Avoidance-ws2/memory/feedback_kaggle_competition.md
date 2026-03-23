---
name: Kaggle competition_sources required for RTX Pro 6000
description: Never remove competition_sources from kernel-metadata.json - needed for RTX Pro 6000 GPU access
type: feedback
---

Do NOT remove or modify `competition_sources: ["nvidia-nemotron-model-reasoning-challenge"]` in Kaggle kernel-metadata.json files.

**Why:** The RTX Pro 6000 Blackwell GPU is only available through this competition. Removing it blocks GPU access.

**How to apply:** When editing any `kernel-metadata.json` in the `kaggle/` folder, always preserve the `competition_sources` field.
