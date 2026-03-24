import streamlit as st
import pandas as pd
import plotly.express as px
import torch

# Import our custom engines
from src.model_loader import ModelLoader
from src.samplers import LogitSampler
from src.advanced_decoding import AdvancedDecoders

# --- Page Config ---
st.set_page_config(page_title="Logit Dynamics", layout="wide", page_icon="🧠")

st.title("🧠 Logit Dynamics: Advanced Inference Control")
st.markdown("Analyze how **Top-K**, **Top-P**, and **Contrastive Layer Decoding (DoLa)** alter the probability landscape to control text generation and mitigate hallucinations.")

# --- Load Model & Engines ---
# Using gpt2 by default. Upgrade to gpt2-medium for better DoLa results if your RAM permits.
model_name = "gpt2" 
model, tokenizer = ModelLoader.load_model(model_name)
device = ModelLoader.get_device()

# Initialize the Math Engines
sampler = LogitSampler(model, tokenizer, device)
adv_decoder = AdvancedDecoders(model, tokenizer, device)

# --- Sidebar Controls ---
st.sidebar.header("Decoding Parameters")
strategy = st.sidebar.radio(
    "Sampling Strategy", 
    ["Greedy/Raw", "Top-K", "Top-P (Nucleus)", "DoLa (Contrastive)"]
)

# Dynamic Sidebar UI based on selected strategy
temperature = 1.0
k_value = 50
p_value = 0.9
alpha_value = 1.0

if strategy in ["Greedy/Raw", "Top-K", "Top-P (Nucleus)"]:
    temperature = st.sidebar.slider("Temperature (Entropy)", 0.01, 2.0, 1.0, 0.05)

if strategy == "Top-K":
    k_value = st.sidebar.slider("Top-K Count", 1, 100, 50)
elif strategy == "Top-P (Nucleus)":
    p_value = st.sidebar.slider("Top-P Mass", 0.1, 0.99, 0.9)
elif strategy == "DoLa (Contrastive)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("DoLa Settings")
    alpha_value = st.sidebar.slider("DoLa Alpha (Penalty Strength)", 0.1, 2.0, 1.0, 0.1)
    st.sidebar.info("Higher Alpha penalizes lower-layer syntax (generic words) more aggressively to amplify the factual signal.")

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Sequence")
    input_text = st.text_area(
        "Context:", 
        "The author of 'Pride and Prejudice' is Jane", 
        height=100
    )
    
    if st.button("Analyze Next Token", type="primary"):
        # 1. Route to the correct Mathematical Engine
        if strategy == "DoLa (Contrastive)":
            # Phase 2: Intercept hidden layers and contrast them
            probs, dola_logits, mature_probs = adv_decoder.get_dola_logits(input_text, alpha=alpha_value)
        else:
            # Phase 1: Standard Output Layer Logits
            probs, raw_logits = sampler.get_next_token_prob_distribution(input_text, temperature)
            
            if strategy == "Top-K":
                probs = sampler.apply_top_k(probs, k=k_value)
            elif strategy == "Top-P (Nucleus)":
                probs = sampler.apply_top_p(probs, p=p_value)
        
        # 2. Get Top 15 Candidates for Visualization
        top_probs, top_indices = torch.topk(probs, 15)
        
        # Decode tokens (cleaning up GPT-2's special 'Ġ' space character)
        top_tokens = [tokenizer.decode([idx]).replace("Ġ", " ") for idx in top_indices]
        
        # 3. Pick the winner (Greedy selection from our engineered distribution)
        winner_idx = torch.argmax(probs).item()
        winner_token = tokenizer.decode([winner_idx]).replace("Ġ", " ")
        
        st.success(f"**Predicted Next Token:** `{winner_token}`")

        # 4. Prepare Visualization Data
        df = pd.DataFrame({
            'Token': top_tokens,
            'Probability': top_probs.tolist()
        })
        
        # Highlight the winning token in a different color for the chart
        df['Status'] = ['Winner' if t == winner_token else 'Candidate' for t in df['Token']]

        # 5. Plotly Chart & Metrics
        with col2:
            st.subheader("Probability Distribution")
            fig = px.bar(
                df, x='Token', y='Probability',
                color='Status',
                title=f"Next Token Probabilities ({strategy})",
                color_discrete_map={'Winner': '#FF4B4B', 'Candidate': '#636EFA'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metric: Shannon Entropy (Uncertainty)
            # Add a tiny epsilon (1e-9) to prevent log(0) errors
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            
            st.metric(
                label="Shannon Entropy ($\mathcal{H}$)", 
                value=f"{entropy:.4f}", 
                help="Measures model uncertainty. High entropy means the model is guessing between many options (high hallucination risk). Low entropy means it is highly confident."
            )