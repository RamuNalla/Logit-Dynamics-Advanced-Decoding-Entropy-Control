import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from src.model_loader import ModelLoader
from src.samplers import LogitSampler

# --- Page Config ---
st.set_page_config(page_title="Logit Dynamics", layout="wide", page_icon="🧠")

st.title("🧠 Logit Dynamics: Phase 1")
st.markdown("### The Probability Landscape of GPT-2")

# --- Load Model ---
model_name = "gpt2" # Change to 'gpt2-medium' for better results if your RAM allows
model, tokenizer = ModelLoader.load_model(model_name)
device = ModelLoader.get_device()
sampler = LogitSampler(model, tokenizer, device)

# --- Sidebar Controls ---
st.sidebar.header("Decoding Parameters")
temperature = st.sidebar.slider("Temperature (Entropy)", 0.01, 2.0, 1.0, 0.05)

strategy = st.sidebar.radio("Sampling Strategy", ["Greedy/Raw", "Top-K", "Top-P (Nucleus)"])

k_value = 0
p_value = 0

if strategy == "Top-K":
    k_value = st.sidebar.slider("Top-K Count", 1, 100, 50)
elif strategy == "Top-P (Nucleus)":
    p_value = st.sidebar.slider("Top-P Mass", 0.1, 0.99, 0.9)

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Sequence")
    input_text = st.text_area("Context:", "The quick brown fox jumps over the", height=100)
    
    if st.button("Analyze Next Token", type="primary"):
        # 1. Get Raw Distribution
        probs, _ = sampler.get_next_token_prob_distribution(input_text, temperature)
        
        # 2. Apply Strategy
        if strategy == "Top-K":
            probs = sampler.apply_top_k(probs, k=k_value)
        elif strategy == "Top-P (Nucleus)":
            probs = sampler.apply_top_p(probs, p=p_value)
        
        # 3. Get Top Candidates for Visualization
        # We grab the top 15 tokens to show the "head" of the distribution
        top_probs, top_indices = torch.topk(probs, 15)
        
        # Decode tokens (handle special chars)
        top_tokens = [tokenizer.decode([idx]).replace("Ġ", " ") for idx in top_indices]
        
        # 4. Visualization Data
        df = pd.DataFrame({
            'Token': top_tokens,
            'Probability': top_probs.tolist()
        })
        
        # Pick the winner (Greedy selection from the modified distribution for demo)
        winner_idx = torch.argmax(probs).item()
        winner_token = tokenizer.decode([winner_idx])
        
        st.success(f"**Predicted Next Token:** `{winner_token}`")

        # 5. Plot
        with col2:
            st.subheader("Probability Distribution")
            fig = px.bar(
                df, x='Token', y='Probability',
                title=f"Next Token Probabilities ({strategy})",
                color='Probability',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metric: Entropy (Uncertainty)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            st.metric("Shannon Entropy", f"{entropy:.4f}", help="Higher = More Uncertain")