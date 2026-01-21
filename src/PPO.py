class PPOTrainer:
    """PPO-style trainer for fine-tuning ESM-2 with LoRA"""
    
    def __init__(
        self, 
        model, 
        ref_model,
        generator, 
        reward_calculator, 
        config
    ):
        self.model = model
        self.ref_model = ref_model
        self.generator = generator
        self.reward_calculator = reward_calculator
        self.config = config
        self.device = next(model.parameters()).device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon
        )
        
        
        self.scheduler = None
        
       
        self.metrics = {
            'rewards': [],
            'kl_divergence': [],
            'policy_loss': [],
            'total_loss': [],
            'avg_sequence_length': [],
            'stability_rewards': [],
            'diversity_rewards': [],
            'constraint_rewards': []
        }
        
    def train(self, num_epochs: int):
        total_steps = num_epochs * (100 // self.config.batch_size)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        print("\n" + "="*80)
        print("Starting RL Training")
        print("="*80)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Total training steps: {total_steps}")
        print("="*80 + "\n")
        
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_metrics = {key: [] for key in self.metrics.keys()}
            
            num_batches = 100 // self.config.batch_size
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx in pbar:
                self.model.eval()
                sequences, _, sequences_str = self.generator.generate_sequences(
                    num_sequences=self.config.num_sequences_per_batch,
                    return_log_probs=False  
                )
                
                
                rewards, reward_components = self.reward_calculator.compute_rewards(sequences_str)
                rewards = rewards.to(self.device)
                
                
                self.model.train()
                outputs = self.model(sequences)
                train_log_probs = self._compute_log_probs(outputs.logits, sequences)
                
                # Compute reference log probs (for KL penalty) - no gradients
                with torch.no_grad():
                    ref_outputs = self.ref_model(sequences)
                    ref_log_probs = self._compute_log_probs(ref_outputs.logits, sequences)
                
                
                kl_div = (train_log_probs.detach() - ref_log_probs).mean()
                
                
                policy_loss = -(train_log_probs * rewards.unsqueeze(1).detach()).mean()
                kl_penalty = self.config.kl_coef * (train_log_probs - ref_log_probs.detach()).mean()
                loss = policy_loss + kl_penalty
                
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                
                epoch_metrics['rewards'].append(rewards.mean().item())
                epoch_metrics['kl_divergence'].append(kl_div.item())
                epoch_metrics['policy_loss'].append(policy_loss.item())
                epoch_metrics['total_loss'].append(loss.item() * self.config.gradient_accumulation_steps)
                epoch_metrics['avg_sequence_length'].append(np.mean([len(s) for s in sequences_str]))
                epoch_metrics['stability_rewards'].append(reward_components['stability'].mean().item())
                epoch_metrics['diversity_rewards'].append(reward_components['diversity'].mean().item())
                epoch_metrics['constraint_rewards'].append(reward_components['constraint'].mean().item())
                
                
                pbar.set_postfix({
                    'reward': f"{rewards.mean().item():.3f}",
                    'loss': f"{loss.item():.3f}",
                    'kl': f"{kl_div.item():.3f}"
                })
                
                
                if self.config.use_wandb and global_step % self.config.log_interval == 0:
                    wandb.log({
                        'reward': rewards.mean().item(),
                        'policy_loss': policy_loss.item(),
                        'kl_divergence': kl_div.item(),
                        'total_loss': loss.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'epoch': epoch
                    }, step=global_step)
                
                
                self.reward_calculator.update_references(sequences_str)
            
            
            for key in self.metrics.keys():
                self.metrics[key].append(np.mean(epoch_metrics[key]))
            
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Avg Reward: {self.metrics['rewards'][-1]:.4f}")
            print(f"  Avg Loss: {self.metrics['total_loss'][-1]:.4f}")
            print(f"  Avg KL: {self.metrics['kl_divergence'][-1]:.4f}")
            print(f"  Stability: {self.metrics['stability_rewards'][-1]:.4f}")
            print(f"  Diversity: {self.metrics['diversity_rewards'][-1]:.4f}")
            print(f"  Constraint: {self.metrics['constraint_rewards'][-1]:.4f}")
            
            
            print(f"\n  Sample sequences:")
            sample_seqs = sequences_str[:3]
            for i, seq in enumerate(sample_seqs):
                print(f"    {i+1}. {seq}")
        
        print("\n Training completed!")
        return self.metrics
    
    def _compute_log_probs(self, logits: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for generated sequences"""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = sequences[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return selected_log_probs


trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    generator=generator,
    reward_calculator=reward_calculator,
    config=config
)