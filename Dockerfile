FROM python:3.10-slim

WORKDIR /app

# Otimização e instalações básicas 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia os requirements e instala
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download prévio das dependências do NLTK usadas no tokenizer
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copia a aplicação inteira
COPY app /app/app
COPY notebooks /app/notebooks

# Expor porta Streamlit
EXPOSE 8501

# Endpoint do container para executar o servidor streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
