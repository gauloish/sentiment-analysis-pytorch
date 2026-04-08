import streamlit as st
import torch
import time
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from model import NBoWModel
from utils import load_vocab, predict_sentiment

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Analisador de Sentimentos IMDB",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- ESTILIZAÇÃO CUSTOMIZADA (CSS Premium) ---
st.markdown("""
<style>
/* Fundo dinâmico da página principal */
.stApp {
    background: linear-gradient(135deg, #090e17 0%, #171d2b 100%);
    color: #e2e8f0;
}

/* Modificadores de cabeçalho global */
h1 {
    font-family: 'Inter', sans-serif;
    color: #ffffff;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0rem;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}

/* Text area (Input de texto) estilizada */
div.stTextArea > div > div > textarea {
    background-color: rgba(30, 41, 59, 0.7);
    color: #f1f5f9;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 15px;
    font-size: 1.05rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

div.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
}

/* Botão principal (Analyze) */
div.stButton > button {
    background: linear-gradient(to right, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    width: 100%;
    margin-top: 1rem;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.4);
    color: white;
}

/* Container de Resultados Positivos e Negativos */
.st-sentiment-positive {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
    border-left: 5px solid #10b981;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 2rem;
    animation: fadeIn 0.8s ease-out forwards;
}

.st-sentiment-negative {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
    border-left: 5px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 2rem;
    animation: fadeIn 0.8s ease-out forwards;
}

.st-sentiment-invalid {
    background: linear-gradient(135deg, rgba(100, 116, 139, 0.1), rgba(100, 116, 139, 0.05));
    border-left: 5px solid #64748b;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 2rem;
}

/* Título de resultado do sentimento */
.sentiment-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.positive-text { color: #34d399; }
.negative-text { color: #f87171; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_assets():
    base_dir = Path(__file__).parent
    vocab_path = base_dir / "assets" / "vocab.json"
    model_path = base_dir / "assets" / "nbow_model.pt"
    
    # 1. Obter vocabulario
    vocab = load_vocab(vocab_path)
    
    # 2. Inicializar os parâmetros conforme o notebook: dim=64, out=1
    model = NBoWModel(vocab_size=len(vocab), embedding_dim=64, output_dim=1, pad_idx=vocab["<pad>"])
    
    # 3. Carregar os pesos
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model, vocab

try:
    model, vocab = load_assets()
except Exception as e:
    st.error(f"Erro ao carregar o modelo ou vocabulário. Verifique os arquivos na pasta 'assets'. Detalhe: {e}")
    st.stop()


# --- FRONTEND E INTERAÇÃO ---
st.markdown("<h1>Analisador de Avaliações IMDB</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Escreva sua avaliação de filme abaixo. Nossa modelo de Inteligência Artificial preverá o sentimento e de quebra, você verá como isso acontece nos bastidores de Machine Learning.</p>", unsafe_allow_html=True)

user_input = st.text_area(" Digite sua avaliação (em inglês):", placeholder="ex: This movie was absolutely phenomenal! The acting and soundtrack left me breathless...", height=180)

if st.button("Analisar Sentimento", key="evaluate_btn"):
    if user_input.strip() == "":
        st.warning("Por favor, digite algum texto antes de analisar.")
    else:
        st.write("---")
        st.markdown("### Processando Pipeline de Machine Learning")
        
        # UI Placeholder para o Pipeline Visual (Stepper)
        pipeline_ui = st.empty()
        
        def render_pipeline(step_num):
            colors = ["#3b82f6" if step_num >= i else "#1e293b" for i in range(1, 6)]
            text_colors = ["#ffffff" if step_num >= i else "#64748b" for i in range(1, 6)]
            
            pipeline_ui.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:stretch; margin-bottom:20px; font-family:sans-serif; gap: 5px;">
                <div style="flex:1; text-align:center; padding:15px 5px; background-color:{colors[0]}; color:{text_colors[0]}; border-radius:8px; border: 1px solid #334155; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-weight:700; margin-bottom:5px;">1. Recepcao</div>
                    <div style="font-size:11px; opacity:0.85;">Validando texto inicial</div>
                </div>
                <div style="display:flex; align-items:center; font-size:16px; color:#64748b;">➔</div>
                <div style="flex:1; text-align:center; padding:15px 5px; background-color:{colors[1]}; color:{text_colors[1]}; border-radius:8px; border: 1px solid #334155; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-weight:700; margin-bottom:5px;">2. Tokenizacao</div>
                    <div style="font-size:11px; opacity:0.85;">Fracionando frase lida</div>
                </div>
                <div style="display:flex; align-items:center; font-size:16px; color:#64748b;">➔</div>
                <div style="flex:1; text-align:center; padding:15px 5px; background-color:{colors[2]}; color:{text_colors[2]}; border-radius:8px; border: 1px solid #334155; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-weight:700; margin-bottom:5px;">3. Embeddings</div>
                    <div style="font-size:11px; opacity:0.85;">Transformando em Tensores</div>
                </div>
                <div style="display:flex; align-items:center; font-size:16px; color:#64748b;">➔</div>
                <div style="flex:1; text-align:center; padding:15px 5px; background-color:{colors[3]}; color:{text_colors[3]}; border-radius:8px; border: 1px solid #334155; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-weight:700; margin-bottom:5px;">4. NBoW Model</div>
                    <div style="font-size:11px; opacity:0.85;">Inferencia Matematica</div>
                </div>
                <div style="display:flex; align-items:center; font-size:16px; color:#64748b;">➔</div>
                <div style="flex:1; text-align:center; padding:15px 5px; background-color:{colors[4]}; color:{text_colors[4]}; border-radius:8px; border: 1px solid #334155; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-weight:700; margin-bottom:5px;">5. Classificacao</div>
                    <div style="font-size:11px; opacity:0.85;">Ativacao e Resultado</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        render_pipeline(0)
        
        sentiment, probability, tokens, indices = predict_sentiment(model, user_input, vocab)
        
        # Animando o Pipeline (MUITO MAIS LENTO PARA EFEITO VISUAL)
        time.sleep(1.0)
        render_pipeline(1)
        time.sleep(1.2)
        render_pipeline(2)
        time.sleep(1.2)
        render_pipeline(3)
        time.sleep(1.5)
        render_pipeline(4)
        time.sleep(1.5)
        render_pipeline(5)
        time.sleep(0.5)
            
        st.write("---")
        
        # Formatando percentual de confiança
        percent = (probability if sentiment == "Positivo" else (1 - probability)) * 100
        
        # TÍTULO GIGANTE COM RESULTADO FINAL
        if sentiment == "Positivo":
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.02)); border-left: 6px solid #10b981; padding: 25px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <h2 style="color: #10b981; margin-top: 0; margin-bottom: 8px; font-family: sans-serif; font-weight: 800; letter-spacing: -0.5px;">VEREDITO DA IA: AVALIACAO POSITIVA</h2>
                <div style="color: #e2e8f0; font-size: 1.15rem; font-family: sans-serif; opacity: 0.9;">O modelo analisou os padroes numericos e determinou que este texto expressa um sentimento favoravel ao filme.</div>
            </div>
            """, unsafe_allow_html=True)
        elif sentiment == "Negativo":
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.02)); border-left: 6px solid #ef4444; padding: 25px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <h2 style="color: #ef4444; margin-top: 0; margin-bottom: 8px; font-family: sans-serif; font-weight: 800; letter-spacing: -0.5px;">VEREDITO DA IA: AVALIACAO NEGATIVA</h2>
                <div style="color: #e2e8f0; font-size: 1.15rem; font-family: sans-serif; opacity: 0.9;">O modelo analisou os padroes numericos e determinou que este texto expressa avaliacao negativa ou criticas ao filme.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, rgba(100, 116, 139, 0.15), rgba(100, 116, 139, 0.02)); border-left: 6px solid #64748b; padding: 25px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <h2 style="color: #94a3b8; margin-top: 0; margin-bottom: 8px; font-family: sans-serif; font-weight: 800; letter-spacing: -0.5px;">TEXTO NAO ENCONTRADO</h2>
                <div style="color: #e2e8f0; font-size: 1.15rem; font-family: sans-serif; opacity: 0.9;">A rede neural nao identificou palavras do vocabulario em ingles neste laco para dar um veredito.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Highlight de Palavras no Texto Original
        st.markdown("### Identificacao Visual de Padroes")
        st.caption("Visao espacial da maquina sobre os tokens. Fundo vermelho = Palavra (ID) desconhecida pelo modelo.")
        
        unk_id = vocab.get("<unk>")
        highlighted_text = []
        for word, idx in zip(tokens, indices):
            if idx == unk_id:
                highlighted_text.append(f"<span style='background-color:#ef4444; color:white; padding:2px 6px; border-radius:4px; font-weight:bold;' title='Palavra Desconhecida'>{word}</span>")
            else:
                highlighted_text.append(f"<span style='color:#10b981; padding:2px;' title='ID: {idx}'>{word}</span>")
                
        st.markdown("<div style='background-color:rgba(30, 41, 59, 0.3); padding:20px; border-radius:8px; line-height:2; font-size:1.1rem;'>" + " ".join(highlighted_text) + "</div>", unsafe_allow_html=True)
        
        st.write("")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Matriz de Tensores")
            st.caption("Tabela exata do tensor subjacente:")
            df_matrix = pd.DataFrame({
                'Token (Palavra)': tokens,
                'ID Vocabulario': indices
            })
            st.dataframe(df_matrix, use_container_width=True, height=250)
            
        with col2:
            st.markdown("### Resultado de Inferencia")
            # Velocímetro de Confiança (Corrigido para exibir a Confiança final e não a Probabilidade crua)
            gauge_color = "#10b981" if sentiment == "Positivo" else "#ef4444"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percent,
                number={'suffix': "%", 'font': {'color': "#e2e8f0"}},
                title={'text': "Nivel de Confianca da IA", 'font': {'color': "#94a3b8", 'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "rgba(30,41,59,0.5)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 100], 'color': "rgba(0, 0, 0, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': percent
                    }
                }
            ))
            fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e2e8f0"})
            st.plotly_chart(fig, use_container_width=True)
