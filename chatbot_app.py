# chatbot_app.py
# Chatbot Jadwal Film Bioskop dengan Streamlit

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import pandas as pd
import nltk
import os
from nltk.stem.porter import PorterStemmer

import os
import nltk

# Setup direktori custom untuk nltk data (agar aman di cloud)
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Download semua resources yang dibutuhkan
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("punkt_tab", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)

# Daftarkan path ke nltk
nltk.data.path.append(nltk_data_dir)

# --- PASTIKAN DATA NLTK 'punkt' TERSEDIA ---
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)

nltk.data.path.append(NLTK_DATA_PATH)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)

# --- STEMMER CACHED ---
@st.cache_resource
def get_stemmer():
    return PorterStemmer()

stemmer = get_stemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# --- LOAD MODEL DAN DATA CHATBOT ---
@st.cache_resource
def load_model_and_data():
    data = torch.load("data.pth")
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self.l3(out)
            return out

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()
    return model, input_size, hidden_size, output_size, all_words, tags

model, input_size, hidden_size, output_size, all_words, tags = load_model_and_data()

# --- LOAD DATASET BIOSKOP ---
@st.cache_data
def load_jadwal_data():
    return pd.read_csv("dataset_jadwal_bioskop.csv")

jadwal_df = load_jadwal_data()

# --- RESPON CHATBOT BERDASARKAN INTENT & ENTITY ---
def get_response(intent, entity):
    if intent == "jadwal_hari_ini":
        hari_ini = pd.Timestamp.today().day_name()
        result = jadwal_df[jadwal_df['hari'].str.lower() == hari_ini.lower()]
    elif intent == "jadwal_hari_tertentu" and entity.get("hari"):
        result = jadwal_df[jadwal_df['hari'].str.lower() == entity["hari"].lower()]
    elif intent == "film_berdasarkan_genre" and entity.get("genre"):
        result = jadwal_df[jadwal_df['genre'].str.lower().str.contains(entity["genre"].lower())]
    elif intent == "film_berdasarkan_lokasi" and entity.get("lokasi"):
        result = jadwal_df[jadwal_df['lokasi'].str.lower().str.contains(entity["lokasi"].lower())]
    else:
        return "Maaf, saya belum dapat menemukan informasi yang sesuai."

    if not result.empty:
        list_film = result.apply(
            lambda row: f"{row['hari']}: {row['judul_film']} (Jam {row['jam_mulai']}–{row['jam_selesai']}, Genre: {row['genre']}, Lokasi: {row['lokasi']})",
            axis=1
        ).tolist()
        return "Berikut jadwal filmnya:\n- " + "\n- ".join(list_film)
    else:
        return "Tidak ditemukan jadwal film yang sesuai."

# --- EKSTRAKSI ENTITY DARI INPUT USER ---
def extract_entity(text):
    hari_list = ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
    entity = {}

    for h in hari_list:
        if h in text.lower():
            entity["hari"] = h
            break

    genre_list = jadwal_df["genre"].dropna().unique()
    for g in genre_list:
        if g.lower() in text.lower():
            entity["genre"] = g
            break

    lokasi_list = jadwal_df["lokasi"].dropna().unique()
    for l in lokasi_list:
        if l.lower() in text.lower():
            entity["lokasi"] = l
            break

    return entity

# --- KLASIFIKASI INTENT USER ---
def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)
    model.eval()
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.40:
        return tag
    else:
        return "unknown"

# --- STREAMLIT APP ---
st.title("🎬 Chatbot Jadwal Film Bioskop 2025")
st.markdown("Tanyakan tentang jadwal film berdasarkan hari, genre, atau lokasi.")

user_input = st.text_input("Tanyakan sesuatu:", "Jadwal bioskop hari Senin apa saja?")

if st.button("Tanya"):
    intent = predict_class(user_input)
    entity = extract_entity(user_input)
    response = get_response(intent, entity)
    st.write(response)
