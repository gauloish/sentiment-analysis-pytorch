import json
import torch
import nltk

def load_vocab(vocab_path):
    """Carrega o arquivo json com o vocabulário."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

import re

def tokenizer(text):
    """
    Realiza a tokenização exata simulando o 'basic_english' do torchtext
    (usado para treinamento), garantindo que expressões como 'don\\'t'
    e pontuações chave continuem iguais.
    """
    text = text.lower()
    _patterns = [
        (r'\'', ' \'  '),
        (r'\"', ''),
        (r'\.', ' . '),
        (r'<br \/>', ' '),
        (r',', ' , '),
        (r'\(', ' ( '),
        (r'\)', ' ) '),
        (r'\!', ' ! '),
        (r'\?', ' ? '),
        (r'\;', ' '),
        (r'\:', ' '),
        (r'\s+', ' ')
    ]
    for pattern, repl in _patterns:
        text = re.sub(pattern, repl, text)
    return text.split()

def predict_sentiment(model, text, vocab, device=torch.device('cpu')):
    """
    Transforma o texto em tensores baseado no vocabulário e alimenta
    o modelo NBoW para predizer o sentimento (Positivo ou Negativo).
    """
    model.eval()

    tokens = tokenizer(text)
    indices = [vocab.get(word, vocab.get("<unk>")) for word in tokens]

    # Se a frase for inválida ou vazia de palavras do nosso vocab
    if len(indices) == 0:
        return "Inválido", 0.5, tokens, indices

    tensor = torch.LongTensor(indices).to(device)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        raw_predict = model(tensor).squeeze(1)
        prob = torch.sigmoid(raw_predict).item()

    sentiment = "Positivo" if prob >= 0.5 else "Negativo"

    return sentiment, prob, tokens, indices
