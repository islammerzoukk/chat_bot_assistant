import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from fonctionnement import words, classes, donnees, lemm

training = []
output_empty = [0] * len(classes)  # vecteur de sortie initialisé à 0

# créer les entrées et sorties du modèle
for d in donnees:
    bag = []
    pattern_words = [lemm.lemmatize(w.lower()) for w in d[0]]
    
    # créer le "bag of words" : 1 si le mot est présent
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    
    output_row = list(output_empty)
    output_row[classes.index(d[1])] = 1  # positionner 1 sur la classe correspondante
    
    training.append([bag, output_row])

# convertir en numpy array pour Keras
training = np.array(training, dtype=object)
train_x = np.array(list(training[:,0]))
train_y = np.array(list(training[:,1]))

# création du modèle
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# compilation
optimizer = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# entraînement
hist = model.fit(x=train_x, y=train_y, epochs=200, batch_size=5, verbose=1)

# sauvegarde du modèle et des données
model.save("chatbot_model.h5")
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

print("✅ Entraînement terminé et modèle sauvegardé !")
