import streamlit  as st
import pickle
import json
import random
import numpy as np
import tensorflow as tf

# Charger le modèle et les données
model = tf.keras.models.load_model("src/chatbot_model.h5")
words = pickle.load(open("src/words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

with open("src/data.json", encoding="utf-8") as file:
    intents = json.load(file)

# Prétraitement du texte utilisateur
def clean_sentence(sentence):
    from nltk.stem import WordNetLemmatizer
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    if not ints:
        return "Je n'ai pas compris, peux-tu reformuler ?"
    tag = ints[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Désolé, je n’ai pas trouvé de réponse."

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Chatbot IA", page_icon="🤖")

st.title("🤖 Mon Chatbot IA")

# Initialiser historique si inexistant
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Afficher l’historique
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrée utilisateur
if user_input := st.chat_input("Écris un message..."):
    # Ajouter le message utilisateur
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Générer la réponse
    ints = predict_class(user_input)
    response = get_response(ints, intents)

    # Ajouter la réponse du bot
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Afficher en direct
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
