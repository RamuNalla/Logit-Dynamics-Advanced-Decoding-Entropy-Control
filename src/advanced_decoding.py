import torch
import torch.nn.functional as F

class AdvancedDecoders:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def get_dola_logits(self, input_text, mature_layer=None, premature_layer=None, alpha=1.0):
        """
        Extracts hidden states and calculates the contrastive DoLa logits.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # 1. Forward pass requesting ALL hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)