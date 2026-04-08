import streamlit as st
import torch
from pathlib import Path

from model import NBoWModel
from utils import load_vocab, predict_sentiment
from components import load_css, render_header, render_result

st.set_page_config(page_title="Analisador de Sentimentos IMDB", layout="centered", initial_sidebar_state="collapsed")
load_css()

@st.cache_resource
def load_assets():
    base_dir = Path(__file__).parent
    vocab_path = base_dir / "assets" / "vocab.json"
    model_path = base_dir / "assets" / "nbow_model.pt"
    
    vocab = load_vocab(vocab_path)
    model = NBoWModel(vocab_size=len(vocab), embedding_dim=64, output_dim=1, pad_idx=vocab["<pad>"])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model, vocab

try:
    model, vocab = load_assets()
except Exception as e:
    st.error(f"Erro ao carregar o modelo ou vocabulário. Detalhe: {e}")
    st.stop()

render_header()
user_input = st.text_area(" Digite sua avaliação (em inglês):", placeholder="ex: This movie was absolutely phenomenal...", height=150)

if st.button("Analisar Sentimento"):
    if user_input.strip() == "":
        st.warning("Por favor, digite algum texto antes de analisar.")
    else:
        st.write("---")
        sentiment, probability = predict_sentiment(model, user_input, vocab)
        
        if sentiment == "Inválido":
            st.warning("Texto Inválido. Não há palavras reconhecidas no vocabulário.")
        else:
            percent = (probability if sentiment == "Positivo" else (1 - probability)) * 100
            render_result(sentiment, percent)
