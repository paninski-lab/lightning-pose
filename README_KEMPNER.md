# Reproducing Multiview Training on Kempner Cluster (A100)

## 1. Installation
Run the setup script to create the environment `lp_repro` with the correct CUDA/DALI versions.

\`\`\`bash
source setup_repro_env.sh
\`\`\`

## 2. Configuration
We use a custom config: `configs/config_kempner_repro.yaml`.
**Critical settings for A100 stability:**
* **DALI Seq Length:** 16 (Prevents OOM)
* **Batch Size:** 1 (Prevents OOM)
* **Image Dims:** 512x512 (Square required for ViT backbone)

## 3. Running Training
Activate the environment and run. Note that we must export `PYTHONNOUSERSITE=1` to prevent local package conflicts.

\`\`\`bash
module load Mambaforge/23.11.0-fasrc01
source activate lp_repro
export PYTHONNOUSERSITE=1 

# Run training
python scripts/train_hydra.py \
    --config-path configs \
    --config-name config_kempner_repro \
    model.model_name=my_multiview_run
\`\`\`
