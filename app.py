import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import time

# Import custom engines
from src.model_loader import ModelLoader
from src.samplers import LogitSampler
from src.advanced_decoding import AdvancedDecoders
from src.metrics import UncertaintyMetrics

# --- Page Config ---
st.set_page_config(page_title="Logit Dynamics", layout="wide", page_icon="🧠")

st.title("🧠 Logit Dynamics: Advanced Inference Control")
st.markdown("Analyze how **Top-K**, **Top-P**, and **Contrastive Layer Decoding (DoLa)** alter the probability landscape and mitigate hallucinations in real-time.")

# --- Load Model & Engines ---
# Using gpt2 for speed. Consider 'gpt2-medium' for more pronounced factual hallucinations to fix.
model_name = "gpt2" 
model, tokenizer = ModelLoader.load_model(model_name)
device = ModelLoader.get_device()

sampler = LogitSampler(model, tokenizer, device)
adv_decoder = AdvancedDecoders(model, tokenizer, device)

# --- Sidebar Controls ---
st.sidebar.header("Decoding Parameters")
strategy = st.sidebar.radio(
    "Sampling Strategy", 
    ["Greedy/Raw", "Top-K", "Top-P (Nucleus)", "DoLa (Contrastive)"]
)

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
    alpha_value = st.sidebar.slider("DoLa Alpha (Penalty)", 0.1, 2.0, 1.0, 0.1)
    st.sidebar.info("Higher Alpha penalizes lower-layer syntax more aggressively.")

# --- Main Layout ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Generation Console")
    input_text = st.text_area(
        "Context:", 
        "The Apollo 11 mission was commanded by", 
        height=100
    )
    
    num_tokens = st.slider("Tokens to Generate", 1, 20, 10)
    
    if st.button("Generate Sequence", type="primary"):
        current_text = input_text
        entropy_history = []
        generated_tokens = []
        
        # Real-time text placeholder
        text_placeholder = st.empty()
        
        # --- Generation Loop ---
        with st.spinner("Decoding logit landscape..."):
            for i in range(num_tokens):
                # 1. Route to Strategy
                if strategy == "DoLa (Contrastive)":
                    probs, _, _ = adv_decoder.get_dola_logits(current_text, alpha=alpha_value)
                else:
                    probs, _ = sampler.get_next_token_prob_distribution(current_text, temperature)
                    if strategy == "Top-K":
                        probs = sampler.apply_top_k(probs, k=k_value)
                    elif strategy == "Top-P (Nucleus)":
                        probs = sampler.apply_top_p(probs, p=p_value)
                
                # 2. Select Token (Greedy selection from the engineered distribution)
                winner_idx = torch.argmax(probs).item()
                raw_token = tokenizer.decode([winner_idx])
                clean_token = raw_token.replace("Ġ", " ") # Clean GPT-2 spacing
                
                # 3. Calculate Shannon Entropy
                entropy = UncertaintyMetrics.calculate_shannon_entropy(probs)
                
                # 4. Update State
                current_text += raw_token
                generated_tokens.append(clean_token)
                entropy_history.append(entropy)
                
                # Update UI
                text_placeholder.markdown(f"**Generated:** `{current_text.replace('Ġ', ' ')}`")
                time.sleep(0.05) # Slight delay for visual effect

        # --- Visualization (Column 2) ---
        with col2:
            st.subheader("Hallucination Monitor (Entropy)")
            
            df_entropy = pd.DataFrame({
                'Step': range(1, num_tokens + 1),
                'Token': generated_tokens,
                'Entropy': entropy_history
            })
            
            # Risk categorization for chart colors
            danger_threshold = 2.5
            df_entropy['Risk'] = ['High (Guessing)' if e > danger_threshold else 'Low (Confident)' for e in df_entropy['Entropy']]
            
            # Plotly Line Chart
            fig = px.line(
                df_entropy, x='Step', y='Entropy', text='Token',
                title=f"Uncertainty over Time ({strategy})",
                markers=True
            )
            
            # Danger Zone Line
            fig.add_hline(
                y=danger_threshold, line_dash="dash", line_color="red", 
                annotation_text="Hallucination Danger Zone", 
                annotation_position="bottom right"
            )
            
            fig.update_traces(textposition="top center")
            fig.update_layout(yaxis_title="Shannon Entropy (bits)", xaxis_title="Token Step")
            st.plotly_chart(fig, use_container_width=True)
            
            # Averages
            avg_entropy = sum(entropy_history) / len(entropy_history)
            st.metric("Average Sequence Entropy", f"{avg_entropy:.2f} bits")