import torch
import torch.nn.functional as F
import numpy as np

class LogitSampler:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)

    def get_next_token_prob_distribution(self, input_text, temperature=1.0):
        """
        1. Tokenizes input.
        2. Passes through model.
        3. Extracts logits of the LAST token.
        4. Applies Temperature scaling.
        5. Returns Softmax probabilities.
        """
        # Encode input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Inference (No Grad for speed)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits of the last token in the sequence
        # Shape: (batch_size, seq_len, vocab_size) -> (vocab_size)
        next_token_logits = outputs.logits[0, -1, :]
        
        # Apply Temperature (Avoid division by zero)
        temperature = max(temperature, 1e-5)
        scaled_logits = next_token_logits / temperature
        
        # Convert to Probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        return probs, scaled_logits

    def apply_top_k(self, probs, k=50):
        """
        Masks all tokens except the Top-K highest probabilities.
        Renormalizes the remaining distribution to sum to 1.
        """
        if k >= len(probs):
            return probs
            
        top_k_probs, top_k_indices = torch.topk(probs, k)
        
        # Create a zero-filled tensor
        masked_probs = torch.zeros_like(probs)
        # Scatter the top-k values back into their original positions
        masked_probs[top_k_indices] = top_k_probs
        
        # Renormalize
        masked_probs = masked_probs / masked_probs.sum()
        return masked_probs

    def apply_top_p(self, probs, p=0.9):
        """
        Nucleus Sampling: Selects smallest set of tokens whose cumulative probability >= p.
        """
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask: remove tokens where cumulative prob > p
        # We shift mask right by 1 to include the token that crossed the threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create the final mask
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        masked_probs = probs.clone()
        masked_probs[indices_to_remove] = 0.0

        # Renormalize
        masked_probs = masked_probs / masked_probs.sum()
        return masked_probs