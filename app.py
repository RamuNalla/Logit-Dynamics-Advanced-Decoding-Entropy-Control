import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import time

# Import custom engines
from src.model_loader import ModelLoader
from src.samplers import LogitSampler
from src.advanced_decoding import AdvancedDecoders
from src.metrics import UncertaintyMetrics

# --- Page Config & Custom CSS ---
st.set_page_config(page_title="Logit Dynamics", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .big-font { font-size:1.2rem !important; line-height: 1.6; }
    .stProgress > div > div > div > div { background-color: #FF4B4B; }
    .metric-card { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Logit Dynamics: The Observatory")
st.markdown("Analyze the inference logit landscape. Use the **Token Inspector** to scrub through time and visualize exactly where hallucinations originate.")

# --- Session State Initialization ---
if 'run_history' not in st.session_state:
    st.session_state.run_history = None
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""

# --- Load Model & Engines ---
@st.cache_resource(show_spinner="Loading Transformer Weights...")
def initialize_system():
    model_name = "gpt2" # Upgrade to 'gpt2-medium' locally for better DoLa results
    model, tokenizer = ModelLoader.load_model(model_name)
    device = ModelLoader.get_device()
    return model, tokenizer, device

model, tokenizer, device = initialize_system()
sampler = LogitSampler(model, tokenizer, device)
adv_decoder = AdvancedDecoders(model, tokenizer, device)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Inference Engine")
    strategy = st.radio(
        "Decoding Strategy", 
        ["Greedy/Raw", "Top-K", "Top-P (Nucleus)", "DoLa (Contrastive)"]
    )

    st.markdown("---")
    st.subheader("Hyperparameters")
    
    temperature = 1.0
    k_value = 50
    p_value = 0.9
    alpha_value = 1.0

    if strategy in ["Greedy/Raw", "Top-K", "Top-P (Nucleus)"]:
        temperature = st.slider("Temperature (Entropy)", 0.01, 2.0, 1.0, 0.05)

    if strategy == "Top-K":
        k_value = st.slider("Top-K Count", 1, 100, 50)
    elif strategy == "Top-P (Nucleus)":
        p_value = st.slider("Top-P Mass", 0.1, 0.99, 0.9)
    elif strategy == "DoLa (Contrastive)":
        alpha_value = st.slider("DoLa Alpha (Penalty)", 0.1, 2.0, 1.0, 0.1)
        st.info("Higher Alpha heavily penalizes generic syntax, forcing factual retrieval.")

# --- Main Layout: Top Section (Input) ---
st.subheader("1. Context Injection")
input_text = st.text_area("Initial Prompt:", "The author of the sci-fi novel 'Dune' is", height=80)

col_gen1, col_gen2 = st.columns([1, 4])
with col_gen1:
    num_tokens = st.number_input("Tokens to Generate", min_value=1, max_value=50, value=15)
with col_gen2:
    st.write("") # Spacing
    st.write("")
    generate_btn = st.button("🚀 Execute Generation", type="primary", use_container_width=True)

# --- Execution Engine ---
if generate_btn:
    current_text = input_text
    history = [] # Temporary list to hold complex state
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_tokens):
        status_text.text(f"Decoding token {i+1}/{num_tokens}...")
        
        # Route Strategy
        if strategy == "DoLa (Contrastive)":
            probs, _, _ = adv_decoder.get_dola_logits(current_text, alpha=alpha_value)
        else:
            probs, _ = sampler.get_next_token_prob_distribution(current_text, temperature)
            if strategy == "Top-K":
                probs = sampler.apply_top_k(probs, k=k_value)
            elif strategy == "Top-P (Nucleus)":
                probs = sampler.apply_top_p(probs, p=p_value)
        
        # Calculate Metrics & Top Tokens BEFORE taking the max (so we see the options)
        entropy = UncertaintyMetrics.calculate_shannon_entropy(probs)
        top_probs, top_indices = torch.topk(probs, 15)
        top_tokens_list = [tokenizer.decode([idx]).replace("Ġ", " ") for idx in top_indices]
        
        # Select Winner
        winner_idx = torch.argmax(probs).item()
        raw_token = tokenizer.decode([winner_idx])
        clean_token = raw_token.replace("Ġ", " ") 
        
        # Save State
        history.append({
            'step': i + 1,
            'token': clean_token,
            'entropy': entropy,
            'top_tokens': top_tokens_list,
            'top_probs': top_probs.tolist()
        })
        
        current_text += raw_token
        progress_bar.progress((i + 1) / num_tokens)
    
    status_text.empty()
    progress_bar.empty()
    
    # Store in session state for interactive scrubbing
    st.session_state.run_history = history
    st.session_state.generated_text = current_text

st.markdown("---")

# --- UI Render (Only runs if we have history in session state) ---
if st.session_state.run_history:
    history = st.session_state.run_history
    
    st.subheader("2. Generation Analysis")
    
    # Render Color-Coded Text
    def get_color(entropy_val):
        if entropy_val < 1.0: return "rgba(0, 255, 0, 0.15)"     # Confident (Green)
        elif entropy_val < 2.5: return "rgba(255, 255, 0, 0.3)"  # Unsure (Yellow)
        else: return "rgba(255, 0, 0, 0.4)"                      # Hallucination Risk (Red)

    html_text = f"<div class='big-font'><b>Context:</b> {input_text} <br><br><b>Generated:</b> "
    for item in history:
        bg_color = get_color(item['entropy'])
        html_text += f"<span style='background-color: {bg_color}; padding: 2px 4px; border-radius: 4px; margin: 0 2px;' title='Entropy: {item['entropy']:.2f}'>{item['token']}</span>"
    html_text += "</div>"
    
    st.markdown(html_text, unsafe_allow_html=True)
    
    # Legend
    st.markdown("<small><b>Heatmap Legend:</b> 🟩 High Confidence | 🟨 Moderate Uncertainty | 🟥 High Hallucination Risk</small>", unsafe_allow_html=True)

    st.write("")
    
    # --- Dual Charts ---
    col_chart1, col_chart2 = st.columns([1.5, 1])
    
    with col_chart1:
        st.markdown("### 📈 Sequence Uncertainty Over Time")
        df_entropy = pd.DataFrame(history)
        
        fig_line = px.line(
            df_entropy, x='step', y='entropy', text='token',
            markers=True, template="plotly_white"
        )
        fig_line.add_hline(y=2.5, line_dash="dash", line_color="red", annotation_text="Danger Zone")
        fig_line.update_traces(textposition="top center", line_color='#636EFA')
        fig_line.update_layout(yaxis_title="Shannon Entropy (bits)", xaxis_title="Token Step", height=400)
        st.plotly_chart(fig_line, use_container_width=True)

    with col_chart2:
        st.markdown("### 🔍 The Token Inspector")
        st.info("Scrub the slider to inspect the exact logit landscape at any specific generation step.")
        
        # The Scrubber
        selected_step = st.slider("Inspect Step:", min_value=1, max_value=len(history), value=1)
        
        # Get data for the selected step
        step_data = history[selected_step - 1]
        
        df_bar = pd.DataFrame({
            'Token': step_data['top_tokens'],
            'Probability': step_data['top_probs']
        })
        df_bar['Status'] = ['Winner' if t == step_data['token'] else 'Candidate' for t in df_bar['Token']]
        
        fig_bar = px.bar(
            df_bar, x='Token', y='Probability', color='Status',
            color_discrete_map={'Winner': '#FF4B4B', 'Candidate': '#A0AEC0'},
            template="plotly_white"
        )
        fig_bar.update_layout(
            title=f"Step {selected_step}: '{step_data['token']}' (Entropy: {step_data['entropy']:.2f})",
            showlegend=False, xaxis_title="", yaxis_title="Probability", height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)