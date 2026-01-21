class RewardCalculator:
    """Compute multiple reward signals for protein sequences"""
    
    def __init__(self, config):
        self.config = config
        
        # Amino acid properties
        self.hydrophobic = set('AILMFWYV')
        self.polar = set('STNQ')
        self.charged = set('DEKR')
        self.positive = set('KR')
        self.negative = set('DE')
        
        
        self.reference_sequences = []
        
    def compute_rewards(
        self, 
        sequences: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined and individual rewards
        
        Returns:
            total_rewards: Combined reward (batch_size,)
            reward_components: Dictionary of individual rewards
        """
        batch_size = len(sequences)
        
        
        stability_rewards = torch.zeros(batch_size)
        diversity_rewards = torch.zeros(batch_size)
        constraint_rewards = torch.zeros(batch_size)
        
        for i, seq in enumerate(sequences):
            
            stability_rewards[i] = self._stability_reward(seq)
            diversity_rewards[i] = self._diversity_reward(seq)
            constraint_rewards[i] = self._constraint_reward(seq)
        
        
        stability_rewards = torch.tanh(stability_rewards)
        diversity_rewards = torch.tanh(diversity_rewards)
        constraint_rewards = torch.tanh(constraint_rewards)
        
        
        total_rewards = (
            self.config.stability_weight * stability_rewards +
            self.config.diversity_weight * diversity_rewards +
            self.config.constraint_weight * constraint_rewards
        )
        
        reward_components = {
            'stability': stability_rewards,
            'diversity': diversity_rewards,
            'constraint': constraint_rewards,
            'total': total_rewards
        }
        
        return total_rewards, reward_components
    
    def _stability_reward(self, seq: str) -> float:
        """
        Stability proxy based on:
        1. Hydrophobic core enrichment
        2. Charge balance
        3. Amino acid composition
        """
        if len(seq) == 0:
            return -1.0
        
        seq = seq.upper()
        score = 0.0
        
       
        hydrophobic_frac = sum(1 for aa in seq if aa in self.hydrophobic) / len(seq)
        score += 1.0 - abs(hydrophobic_frac - 0.35) * 2  # Penalty if far from 35%
        
        
        positive_count = sum(1 for aa in seq if aa in self.positive)
        negative_count = sum(1 for aa in seq if aa in self.negative)
        net_charge = abs(positive_count - negative_count) / len(seq)
        score += 1.0 - net_charge * 2  
        
        
        max_repeat = self._max_consecutive_repeat(seq)
        score -= max_repeat * 0.2  
        
        
        helix_formers = set('AELM')
        sheet_formers = set('VIY')
        ss_frac = sum(1 for aa in seq if aa in helix_formers or aa in sheet_formers) / len(seq)
        score += ss_frac * 0.5
        
        return score
    
    def _diversity_reward(self, seq: str) -> float:
        """
        Diversity reward based on:
        1. Amino acid diversity (Shannon entropy)
        2. Distance from reference sequences
        """
        if len(seq) == 0:
            return -1.0
        
        seq = seq.upper()
        score = 0.0
        
        
        aa_counts = {}
        for aa in seq:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        entropy = 0.0
        for count in aa_counts.values():
            prob = count / len(seq)
            entropy -= prob * np.log2(prob + 1e-10)
        
        
        normalized_entropy = entropy / 4.32
        score += normalized_entropy
        
        
        if self.reference_sequences:
            min_distance = float('inf')
            for ref_seq in self.reference_sequences:
                distance = self._edit_distance(seq, ref_seq)
                min_distance = min(min_distance, distance)
            
            
            normalized_distance = min_distance / max(len(seq), 1)
            score += normalized_distance * 0.5
        else:
            score += 0.5  
        
        return score
    
    def _constraint_reward(self, seq: str) -> float:
        """
        Constraint satisfaction reward:
        1. Length constraints
        2. Forbidden motifs
        3. Required motifs
        """
        if len(seq) == 0:
            return -1.0
        
        seq = seq.upper()
        score = 0.0
        
        
        target_length = (self.config.min_seq_length + self.config.max_seq_length) / 2
        length_penalty = abs(len(seq) - target_length) / target_length
        score += 1.0 - length_penalty
        
        # 2. Forbidden motifs (e.g., glycine-proline repeats)
        forbidden_motifs = ['GGG', 'PPP', 'CCC']
        for motif in forbidden_motifs:
            if motif in seq:
                score -= 0.5
        
        # 3. Desired motifs (e.g., catalytic triads, binding motifs)
        # Example: reward presence of common functional motifs
        desired_motifs = ['RGD', 'KRK']  
        for motif in desired_motifs:
            if motif in seq:
                score += 0.3
        
        
        rare_aas = set('WCM')
        rare_frac = sum(1 for aa in seq if aa in rare_aas) / len(seq)
        if rare_frac > 0.15:  # Penalize if >15% rare amino acids
            score -= (rare_frac - 0.15) * 2
        
        return score
    
    def _max_consecutive_repeat(self, seq: str) -> int:
        """Find longest consecutive repeat of same amino acid"""
        if not seq:
            return 0
        max_len = 1
        current_len = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
        return max_len
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def update_references(self, sequences: List[str]):
        self.reference_sequences.extend(sequences)
        
        if len(self.reference_sequences) > 100:
            self.reference_sequences = self.reference_sequences[-100:]

reward_calculator = RewardCalculator(config)


print("\n" + "="*80)
print("Testing Reward Functions")
print("="*80)

test_sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAK",
    "GGGGGGGGGGGGGGGGGGGG",  # Bad: poly-G
    "AVILMFWYAVILMFWYAVIL"   # Good: hydrophobic rich
]

for seq in test_sequences:
    total_reward, components = reward_calculator.compute_rewards([seq])
    print(f"\nSequence: {seq[:50]}...")
    print(f"  Length: {len(seq)}")
    print(f"  Stability:  {components['stability'].item():6.3f}")
    print(f"  Diversity:  {components['diversity'].item():6.3f}")
    print(f"  Constraint: {components['constraint'].item():6.3f}")
    print(f"  TOTAL:      {components['total'].item():6.3f}")