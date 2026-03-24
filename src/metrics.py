import torch

class UncertaintyMetrics:
    @staticmethod
    def calculate_shannon_entropy(probs, epsilon=1e-9):
        """
        Calculates the Shannon Entropy of a probability distribution.
        High Entropy = Flat Distribution = High Uncertainty / Hallucination Risk.
        Low Entropy = Sharp Distribution = High Confidence.
        
        Args:
            probs (torch.Tensor): The probability distribution (summing to 1).
            epsilon (float): A tiny value to prevent log(0) errors.
        """
        # Ensure we are working with a 1D tensor for a single token
        if probs.dim() > 1:
            probs = probs.squeeze()
            
        entropy = -torch.sum(probs * torch.log(probs + epsilon))
        return entropy.item()

    @staticmethod
    def is_hallucinating(entropy, threshold=2.5):
        """
        A simple heuristic: if entropy crosses a threshold, flag as hallucination risk.
        Note: The ideal threshold depends on the vocabulary size and temperature.
        """
        return entropy > threshold