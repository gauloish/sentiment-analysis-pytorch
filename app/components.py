import streamlit as st
import os

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_header():
    st.markdown("<h1>Analisador de Críticas IMDB</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Escreva sua avaliação de filme abaixo e nossa Inteligência Artificial preverá o sentimento associado.</p>", unsafe_allow_html=True)

def render_result(sentiment, percent):
    if sentiment == "Positivo":
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.02)); border-left: 6px solid #10b981; padding: 25px; border-radius: 8px; margin-bottom: 30px;">
            <h2 style="color: #10b981; margin-top: 0; margin-bottom: 8px;">AVALIACAO POSITIVA</h2>
            <div style="color: #e2e8f0; font-size: 1.15rem;">A IA esta <b>{percent:.1f}%</b> confiante de que esta e uma avaliacao favoravel ao filme.</div>
        </div>
        """, unsafe_allow_html=True)
    elif sentiment == "Negativo":
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.02)); border-left: 6px solid #ef4444; padding: 25px; border-radius: 8px; margin-bottom: 30px;">
            <h2 style="color: #ef4444; margin-top: 0; margin-bottom: 8px;">AVALIACAO NEGATIVA</h2>
            <div style="color: #e2e8f0; font-size: 1.15rem;">A IA esta <b>{percent:.1f}%</b> confiante de que esta e uma critica ou avaliacao negativa.</div>
        </div>
        """, unsafe_allow_html=True)
