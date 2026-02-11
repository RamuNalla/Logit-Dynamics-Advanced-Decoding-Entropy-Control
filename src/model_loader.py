import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class ModelLoader:
    """
    Using 'gpt2' (small) for speed.
    """
    
    @staticmethod
    @st.cache_resource
    def load_model(model_name="gpt2"):
        print(f"Loading {model_name}...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set to evaluation mode (disables dropout, etc.)
        model.eval()
        
        return model, tokenizer

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"