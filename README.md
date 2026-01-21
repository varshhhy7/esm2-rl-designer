# Controllable Protein Design via RL Fine-Tuning of ESM-2

## Overview

This project implements a **Reinforcement Learning (RL)** framework for controllable protein design by fine-tuning Meta's **ESM-2** (Evolutionary Scale Modeling) protein language model using **LoRA** (Low-Rank Adaptation). The approach combines multi-objective reward shaping with policy gradient optimization to generate novel protein sequences with desired structural and functional properties.

### Key Features

- **Parameter-Efficient Fine-Tuning:** LoRA adapters targeting attention modules (Q, K, V) with <1% trainable parameters
- **Multi-Objective Reward Design:** Balancing stability, diversity, and constraint satisfaction
- **Autoregressive Generation:** Custom sampling pipeline with top-k and nucleus (top-p) filtering
- **Real-Time Monitoring:** Weights & Biases (WandB) integration for experiment tracking
- **FP32 Precision:** Full precision training for numerical stability

---

## Architecture & Training Pipeline

### 1. Environment & Hardware Verification

**Purpose:** Ensures GPU availability for efficient tensor operations and gradient calculations.

- **Tool:** `nvidia-smi` (NVIDIA System Management Interface)
- **Check:** Verifies CUDA-enabled device presence and VRAM availability
- **Safety:** Terminates execution if no GPU detected to prevent OOM errors

### 2. Dependency Installation

The project leverages the Hugging Face ecosystem and Meta's ESM tools:

| Library | Function |
|---------|----------|
| **transformers** | Pre-trained ESM-2 model architecture and tokenizers |
| **peft** | LoRA implementation for parameter-efficient fine-tuning |
| **accelerate** | Device placement and distributed training optimizations |
| **fair-esm** | Native Meta AI tools for ESM weight management |
| **wandb** | Experiment tracking and multi-objective reward visualization |

### 3. Global Configuration & Model Setup

#### Model Backbone
- **ESM-2 (650M parameters):** Large-scale protein language model pre-trained on evolutionary protein sequences
- **LoRA Configuration:**
  - Rank (r): 8
  - Alpha: 16
  - Target Modules: Query, Key, Value matrices in attention layers
  - Dropout: 0.05
  - **Result:** ~0.2% trainable parameters

#### Generation Configuration
- **Max Sequence Length:** 64 amino acids
- **Min Sequence Length:** 32 amino acids
- **Temperature:** 1.0 (sampling diversity control)
- **Top-K:** 50 (restrict to top-50 likely tokens)
- **Top-P (Nucleus):** 0.9 (cumulative probability threshold)

#### RL Training Configuration
- **Epochs:** 5
- **Batch Size:** 4
- **Sequences per Batch:** 8
- **Gradient Accumulation Steps:** 4
- **Learning Rate:** 5e-5
- **KL Penalty Coefficient:** 0.1 (controls deviation from reference model)
- **Warmup Steps:** 100

#### Reward Weights
- **Stability Weight:** 1.0 (prioritizes structural integrity)
- **Diversity Weight:** 0.5 (encourages novel sequences)
- **Constraint Weight:** 0.5 (enforces design rules)

### 4. Model Initialization & LoRA Adaptation

#### Dual-Model Architecture

1. **Active Model (Actor):** Base ESM-2 + LoRA adapters
   - Only trainable component
   - Optimized via policy gradients
   - Adapted for desired sequence properties

2. **Reference Model:** Frozen original ESM-2
   - Used to compute KL penalty
   - Ensures biological plausibility
   - Prevents mode collapse

#### Key Optimizations
- **FP32 Precision:** Maintains numerical stability for large embeddings
- **Low CPU Memory Usage:** Efficient weight loading strategy
- **Gradient Checkpointing:** (Optional) Further reduces memory footprint

---

## Core Components

### 5. Autoregressive Protein Generation

Implements custom de novo protein design through iterative token prediction, since ESM-2 is natively a Masked Language Model.

#### Sampling Techniques

**Top-K Filtering:** Restricts sampling to the K most likely amino acids to prevent invalid transitions

**Top-P (Nucleus) Sampling:** Dynamically selects the smallest token set with cumulative probability ≥ P, balancing diversity and validity

**Log-Probability Tracking:** Essential for policy gradient updates, correlates sequence choices with resulting rewards

#### Generation Workflow
1. Initialize with `[CLS]` token
2. Generate logits for next amino acid position
3. Apply temperature, top-k, and top-p filtering
4. Sample next token and append to sequence
5. Repeat until max_length or `[EOS]` reached

#### Valid Amino Acids
Restricted to the 20 standard amino acids:
- **LAGVSERTIDPKQNFYMHWC**
- Special tokens filtered out (CLS, PAD, MASK, EOS)

---

### 6. Reward Calculation

Implements multi-objective reward signal computation balancing three key aspects:

#### Stability Reward
Measures structural integrity through:
- **Hydrophobic Core:** Optimal hydrophobic fraction ~35%
- **Charge Balance:** Minimizes net charge deviation
- **Repetition Penalty:** Discourages long consecutive repeats (>3)
- **Secondary Structure Propensity:** Rewards helix-forming (AELM) and sheet-forming (VIY) amino acids

#### Diversity Reward
Encourages novelty and design space exploration:
- **Shannon Entropy:** Amino acid composition diversity
- **Euclidean Distance:** Separation from reference sequences
- **Normalized Entropy:** Scaled to [0,1] range

#### Constraint Reward
Enforces design rules and biological plausibility:
- **Length Constraints:** Penalizes deviation from target range
- **Forbidden Motifs:** Penalizes GGG, PPP, CCC repeats
- **Desired Motifs:** Rewards RGD, KRK binding motifs
- **Rare AA Control:** Limits W, C, M to <15% of sequence

#### Reward Combination
```
total_reward = 1.0 * tanh(stability) + 0.5 * tanh(diversity) + 0.5 * tanh(constraint)
```

---

### 7. PPO-Style Trainer

Implements Policy Gradient Optimization with KL penalty for controllable fine-tuning.

#### Loss Function
```
policy_loss = -mean(log_prob * reward)
kl_penalty = kl_coef * mean(log_prob_model - log_prob_reference)
total_loss = policy_loss + kl_penalty
```

#### Optimization Strategy
- **Optimizer:** AdamW with ε=1e-8
- **Scheduler:** Cosine annealing with warmup (100 steps)
- **Gradient Clipping:** Max norm = 1.0
- **Gradient Accumulation:** 4 steps for effective batch size = 16

#### Training Loop
For each epoch:
1. Generate protein sequences using current model
2. Compute reward signals (stability, diversity, constraint)
3. Forward pass through model to get log probabilities
4. Compute reference log probabilities (frozen model)
5. Calculate combined loss (policy + KL penalty)
6. Backward pass and optimizer step
7. Log metrics to WandB

#### Tracked Metrics
- Reward (total, stability, diversity, constraint)
- Policy loss and KL divergence
- Total loss
- Average sequence length
- Learning rate

---

## Training Execution

The trainer initializes with:
- **Epochs:** 5
- **Total Steps:** 100 sequences per epoch / 4 batch size = 25 batches
- **Warmup:** 100 steps of linear LR increase
- **Logging Interval:** Every 10 steps to WandB

### Example Training Output
```
Epoch 1/5: 100%
  reward: 0.324
  loss: 0.456
  kl: 0.087

Epoch 1 Summary:
  Avg Reward: 0.3240
  Avg Loss: 0.4563
  Avg KL: 0.0875
  Stability: 0.2156
  Diversity: 0.1984
  Constraint: 0.1859
  
  Sample sequences:
    1. MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSR
    2. LVIGKVTDGAKGIIVVSKDH...
    3. AVILMFWYAVILMFWYAVIL...
```

---

## Project Structure

```
c:\esm2-rl-designer\
├── Controllable Protein Design via RL Fine-Tuning of ESM-2.ipynb
│   ├── 1. Environment & Hardware Verification
│   ├── 2. Dependency Installation
│   ├── 3. Global Configuration & Experiment Setup
│   ├── 4. Model Initialization & LoRA Adaptation
│   ├── 5. Autoregressive Protein Generation
│   ├── 6. Reward Calculation
│   └── 7. PPO-Style Trainer [Current: Cell 18]
└── README.md (this file)
```

---

## Comprehensive Metrics & Tracking

### Core Training Metrics (Per-Epoch & Real-Time)

#### Objective Function Components
| Metric | Description | Formula | Tracked |
|--------|-------------|---------|----------|
| **Total Reward** | Combined multi-objective signal | 1.0×tanh(S) + 0.5×tanh(D) + 0.5×tanh(C) |  Every batch |
| **Stability Reward** | Structural integrity proxy | hydro_frac + charge_balance + ss_propensity - repeats |  Per-epoch avg |
| **Diversity Reward** | Sequence novelty & exploration | shannon_entropy + ref_distance |  Per-epoch avg |
| **Constraint Reward** | Design rule satisfaction | length_penalty + motif_score + rare_aa_control |  Per-epoch avg |

#### Optimization Metrics
| Metric | Description | Range | Tracked |
|--------|-------------|-------|----------|
| **Policy Loss** | -mean(log_prob × reward) | [0, ∞) |  Every batch |
| **KL Divergence** | KL(model \| reference) | [0, ∞) |  Every batch |
| **Total Loss** | policy_loss + 0.1×KL_divergence | [0, ∞) |  Per-epoch avg |
| **Learning Rate** | Cosine annealing with warmup | [0, 5e-5] |  Every batch |
| **Gradient Norm** | Max gradient clipping value | [0, 1.0] |  Per-step |

#### Sequence Properties
| Metric | Description | Target | Tracked |
|--------|-------------|--------|----------|
| **Avg Sequence Length** | Mean length of generated sequences | 48 ± 16 | ✅ Per-epoch |
| **Max Consecutive Repeat** | Longest consecutive AA | < 3 | ✅ Per-sequence |
| **Hydrophobic Fraction** | % hydrophobic AAs | ~35% | ✅ Per-sequence |
| **Charge Balance** | \|positive - negative\| / length | < 0.2 | ✅ Per-sequence |
| **Rare AA Frequency** | (W + C + M) / length | < 15% | ✅ Per-sequence |

### Evaluation Metrics (Post-Training)

#### Per-Sequence Analysis
- **Individual Rewards:** Total, stability, diversity, constraint scores for each generated sequence
- **Length Distribution:** Histogram of all generated sequence lengths (target: 32-64 AA)
- **AA Composition:** Frequency bar chart across all 20 standard amino acids
- **Reward Scatter:** Per-sequence reward visualization with mean/std baselines
- **Component Breakdown:** (S, D, C) contribution to total reward per sequence

#### Outputs Saved
| File | Content | Format |
|------|---------|--------|
| `rl_training_results.png` | 6-panel visualization (see below) | PNG, 300 DPI |
| `reward_distribution.png` | Scatter plot with color-coded rewards | PNG, 300 DPI |
| `generated_sequences.csv` | 20 sequences + all metrics | CSV |
| `training_metrics.csv` | Per-epoch aggregated metrics | CSV |
| `esm2_rl_finetuned/` | Fine-tuned model + LoRA adapters | PyTorch |

### Visualization Panels (rl_training_results.png)

**Panel 1: Average Total Reward (Top-Left)**
- Line plot: Reward trend across epochs
- Markers: One point per epoch
- Interpretation: Upward trend indicates successful policy learning

**Panel 2: Total Loss (Top-Right)**
- Line plot: Combined loss (policy + KL penalty)
- Color: Red for emphasis on optimization
- Interpretation: Downward trend = convergence; sharp drops = learning rate impact

**Panel 3: KL Divergence (Middle-Left)**
- Line plot: Distance from reference model
- Color: Orange
- Interpretation: Lower values prevent mode collapse; high values = divergence risk

**Panel 4: Reward Components Over Time (Middle-Right)**
- Multi-line plot: Stability (○), Diversity (□), Constraint (△)
- Interpretation: Trade-offs between objectives; which component dominates training

**Panel 5: Sequence Length Distribution (Bottom-Left)**
- Histogram: Count vs. sequence length
- Red dashed line: Mean length
- Interpretation: Clustering indicates controlled generation; spread = diversity

**Panel 6: Amino Acid Composition (Bottom-Right)**
- Bar chart: Frequency of each of 20 AAs
- Color: Light coral for visibility
- Interpretation: Composition bias; enrichment of certain AAs due to reward signals

### Additional Visualization: Reward Distribution (reward_distribution.png)

**Scatter Plot Analysis**
- **X-axis:** Sequence index (0-19 generated)
- **Y-axis:** Total reward per sequence
- **Color Map:** Viridis (blue=low reward, yellow=high reward)
- **Red Line:** Mean reward across all 20 sequences
- **Interpretation:** Spread indicates reward variance; tight clustering = consistent generation quality

### WandB Logging

If `use_wandb=true`, the following metrics stream to the WandB dashboard in real-time:

```
wandb.log({
    'reward': float,              # Per-batch mean reward
    'policy_loss': float,         # Per-batch policy loss
    'kl_divergence': float,       # Per-batch KL divergence
    'total_loss': float,          # Per-batch total loss
    'learning_rate': float,       # Current LR from scheduler
    'epoch': int                  # Current epoch number
}, step=global_step)
```

### Metric Summary Report (Printed at Epoch End)

After each epoch:
```
Epoch 1 Summary:
  Avg Reward: 0.3240
  Avg Loss: 0.4563
  Avg KL: 0.0875
  Stability: 0.2156
  Diversity: 0.1984
  Constraint: 0.1859
  
  Sample sequences:
    1. MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSR
    2. LVIGKVTDGAKGIIVVSKDH...
    3. AVILMFWYAVILMFWYAVIL...
```

---

## Work in Progress

**Current Status:** Core training pipeline implemented and tested. The following components are still under development:

1. **Interactive Visualization Tool** (Coming Soon)
   - Real-time WandB dashboard integration
   - Multi-objective trade-off curves
   - Interactive sequence property explorer
   - Model prediction confidence visualization

2. **Advanced Inference & Evaluation**
   - Systematic generation test suite
   - Comparative analysis against baseline sequences
   - Structural property prediction (if ESMFold/OmegaFold available)
   - Binding affinity estimation

3. **Model Checkpointing & Deployment**
   - Best-model checkpoint saving during training
   - LoRA adapter export for inference-only mode
   - Model versioning with WandB artifacts
   - Inference pipeline optimization

4. **Advanced Training Features** (Optional)
   - Multi-GPU distributed training with `torch.nn.DataParallel`
   - Curriculum learning for progressive constraint tightening
   - Ensemble training across multiple random seeds
   - Adaptive reward weighting based on convergence

---

## System Requirements

### Google Colab Free Tier (Primary Target)
- **GPU:** NVIDIA T4 (15 GB VRAM) - Google Colab Free
- **Compute Cores:** 2× Tesla T4 (optional: Colab Pro → A100 or V100)
- **RAM:** 13 GB system RAM
- **Storage:** ~20 GB (ESM-2 weights + outputs)
- **Python:** 3.10+ (Colab default)
- **Runtime:** ~45-60 minutes per 5 epochs on T4


### Memory Optimization for Colab T4
- **Model Size:** ESM-2 (650M) with LoRA = ~3 GB loaded
- **Batch Size:** 4 (reduced from typical 16 for V100)
- **Gradient Accumulation:** 4 steps (effective batch = 16)
- **Precision:** FP32 for stability; FP16 reduces VRAM to ~2 GB (optional)
- **Estimated Runtime:** ~50 min per epoch on T4 GPU

### Installation in Colab

```bash
# Install dependencies
!pip install torch transformers peft accelerate fair-esm wandb matplotlib pandas tqdm

# Verify GPU
!nvidia-smi
```

---

## Hyperparameter Justification

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 8 | Balances expressiveness with parameter efficiency |
| Learning rate | 5e-5 | Conservative for fine-tuning; prevents catastrophic forgetting |
| KL coefficient | 0.1 | Strong KL penalty prevents divergence from base model |
| Stability weight | 1.0 | Primary objective; ensures sequences remain foldable |
| Diversity weight | 0.5 | Secondary; encourages exploration within valid space |
| Batch size | 4 | Memory constraint; paired with gradient accumulation |

---

## References

- **ESM-2:** Lin et al. (2023) "Protein Language Models Enable Learning of Functional Constraints"
- **LoRA:** Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- **PPO:** Schulman et al. (2017) "Proximal Policy Optimization Algorithms"

---

## Experiment Tracking

All experiments are logged to **Weights & Biases (WandB)** under the project `protein-rl-design` with run name `esm2-rl-experiment`.

Key metrics visible in WandB dashboard:
- Real-time reward convergence
- Multi-objective trade-off curves
- Sequence length distribution
- Model parameter statistics

---

**Last Updated:** January 21, 2026  
**Current Progress:** Core training pipeline complete (Cells 1-18) with comprehensive metrics & visualization  
**Status:**  **WORK IN PROGRESS**  
**Tested Environments:** Google Colab Free Tier (T4 GPU)
**Metric Outputs:** 6-panel visualization + CSV exports + WandB streaming
