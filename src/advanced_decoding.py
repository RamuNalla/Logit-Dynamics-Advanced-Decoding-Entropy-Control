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

        # 2. Identify Layers
        # GPT-2 small has 12 layers (0 to 11). 
        # outputs.hidden_states has 13 elements (input embeddings + 12 layers)
        num_layers = self.model.config.n_layer
        if mature_layer is None: 
            mature_layer = num_layers # The final layer
        if premature_layer is None: 
            premature_layer = num_layers // 2 # The middle layer

        # 3. Extract the hidden state for the LAST token from our target layers
        mature_hidden = outputs.hidden_states[mature_layer][0, -1, :]
        premature_hidden = outputs.hidden_states[premature_layer][0, -1, :]    

        # 4. Project hidden states to Vocabulary Logits using the LM Head
        mature_logits = self.model.lm_head(mature_hidden)
        premature_logits = self.model.lm_head(premature_hidden)

        # 5. Apply Contrastive Subtraction
        # We only want to contrast plausible tokens, so we dynamically mask the tail
        mature_probs = F.softmax(mature_logits, dim=-1)

        # Dynamic threshold: Only consider tokens in the Top-K of the mature layer
        top_k_probs, _ = torch.topk(mature_probs, 50)
        threshold = top_k_probs[-1] # The 50th highest probability

        # Log-odds difference: Mature - (Alpha * Premature)
        contrastive_logits = mature_logits - (alpha * premature_logits)

        # Mask out the noise (set tokens below threshold to negative infinity)
        mask = mature_probs < threshold
        contrastive_logits[mask] = -float('inf')

        # Convert the new contrastive logits to probabilities for visualization
        dola_probs = F.softmax(contrastive_logits, dim=-1)

        return dola_probs, contrastive_logits, mature_probs