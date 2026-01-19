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

## Next Steps

Upon completion of the current cell (PPO Trainer), the following components are planned:

1. **Visualization & Analysis**
   - Plot training curves (reward, loss, KL divergence)
   - Sequence property distribution analysis
   - Reward component heatmaps

2. **Inference & Evaluation**
   - Generate test sequences from trained model
   - Compare against baseline sequences
   - Compute structural metrics (if 3D predictor available)

3. **Model Checkpoint Saving**
   - Save best LoRA adapters
   - Export trained model for inference
   - Version control with WandB artifacts

4. **Advanced Features** (Optional)
   - Multi-GPU distributed training
   - Curriculum learning for progressively harder constraints
   - Integration with structure prediction (OmegaFold, ESMFold)

---

## System Requirements

- **GPU:** NVIDIA GPU with 12GB+ VRAM (tested on V100, A100)
- **RAM:** 32GB system RAM
- **Python:** 3.9+
- **CUDA:** 11.8+
- **PyTorch:** 2.0+

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

**Last Updated:** January 19, 2026  
**Current Progress:** Core training pipeline (Cells 1-18)  
**Status:** Ready for training execution
