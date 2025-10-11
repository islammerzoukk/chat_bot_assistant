import json
import nltk
from nltk.stem import WordNetLemmatizer

# initialisation du lemmatizer
lemm = WordNetLemmatizer()

# charger le dataset
with open("src/data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

words = []    # vocabulaire
classes = []  # catégories
donnees = []  # couples (mot_tokenisé, tag)

# tokenization des phrases
for i in data["intents"]:
    for p in i["patterns"]:
        word_list = nltk.word_tokenize(p)   # découper la phrase en mots
        words.extend(word_list)             # ajouter au vocabulaire
        donnees.append((word_list, i["tag"]))
        if i["tag"] not in classes:
            classes.append(i["tag"])

# lemmatisation et nettoyage
words = [lemm.lemmatize(w.lower()) for w in words if w not in ["?", "!", "."]]
words = sorted(list(set(words)))  # retirer les doublons
classes = sorted(list(set(classes)))

# juste pour vérifier
print("Mots:", words)
print("Classes:", classes)
print("Données:", donnees)
