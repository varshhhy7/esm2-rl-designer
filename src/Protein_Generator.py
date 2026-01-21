class ProteinGenerator:
    """Handles autoregressive protein sequence generation"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else None
        
        self.valid_token_ids = list(range(4, 24))  
        
        self.valid_token_ids = torch.tensor(self.valid_token_ids, dtype=torch.long, device=self.device)
        
        print(f"\n{'='*80}")
        print(f"ProteinGenerator initialized")
        print(f"  Total vocabulary size: {len(tokenizer)}")
        print(f"  Valid amino acid tokens: {len(self.valid_token_ids)}")
        print(f"  Valid token IDs: {self.valid_token_ids.tolist()}")
        
        
        valid_tokens_str = [tokenizer.convert_ids_to_tokens(i) for i in self.valid_token_ids.tolist()]
        print(f"  Allowed amino acids: {', '.join(valid_tokens_str)}")
        print(f"{'='*80}\n")
        
    def generate_sequences(
        self, 
        num_sequences: int,
        temperature: float = None,
        max_length: int = None,
        return_log_probs: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
        """Generate protein sequences autoregressively"""
        temperature = temperature or self.config.temperature
        max_length = max_length or self.config.max_seq_length
        
        self.model.eval()
        
       
        sequences = torch.full(
            (num_sequences, 1), 
            self.cls_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        log_probs_list = []
        
        with torch.no_grad():
            for step in range(max_length - 1):
                outputs = self.model(sequences)
                logits = outputs.logits[:, -1, :] 
                
                
                valid_logits = torch.full_like(logits, float('-inf'))
                
                
                valid_logits[:, self.valid_token_ids] = logits[:, self.valid_token_ids]
                
                
                valid_logits = valid_logits / temperature
                
                
                valid_logits = self._top_k_top_p_filtering(
                    valid_logits, 
                    self.config.top_k, 
                    self.config.top_p
                )
                
                
                if torch.isnan(valid_logits).any() or torch.isinf(valid_logits).all():
                    print(f"Warning: Invalid logits detected at step {step}")
                    valid_logits = torch.nan_to_num(valid_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                
                probs = F.softmax(valid_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                
                max_token_id = next_token.max().item()
                if max_token_id >= len(self.tokenizer):
                    raise ValueError(
                        f"Generated invalid token ID {max_token_id} "
                        f"(vocab size: {len(self.tokenizer)})"
                    )
                
                if return_log_probs:
                    log_prob = F.log_softmax(valid_logits, dim=-1)
                    selected_log_probs = log_prob.gather(1, next_token)
                    log_probs_list.append(selected_log_probs)
                
                sequences = torch.cat([sequences, next_token], dim=1)
        
        log_probs = torch.cat(log_probs_list, dim=1) if (return_log_probs and log_probs_list) else None
        
        
        sequences_str = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        return sequences, log_probs, sequences_str
    
    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0):
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        return logits
    

torch.cuda.empty_cache()


generator = ProteinGenerator(model, tokenizer, config)


print("\n" + "="*80)
print("Testing Fixed Generator (Standard Amino Acids Only)")
print("="*80)

try:
    test_sequences, test_log_probs, test_str = generator.generate_sequences(
        num_sequences=3, 
        max_length=20,
        return_log_probs=True
    )
    print(f" Successfully generated {len(test_str)} sequences!")
    print(f"\nGenerated sequences:")
    for i, seq in enumerate(test_str):
        print(f"  {i+1}. {seq} (length: {len(seq)})")
    

    print(f"\nVerifying sequences contain only standard amino acids...")
    standard_aas = set('LAGVSERTIDPKQNFYMHWC')
    all_valid = True
    for seq in test_str:
        seq_chars = set(seq.replace(' ', ''))
        if not seq_chars.issubset(standard_aas):
            print(f"  Found invalid characters: {seq_chars - standard_aas}")
            all_valid = False
    
    if all_valid:
        print(f" All sequences contain only standard amino acids!")
    
    print("="*80)
except Exception as e:
    print(f" Error during generation: {e}")
    import traceback
    traceback.print_exc()
    print("="*80)

print("\n Generator is ready for training!")
