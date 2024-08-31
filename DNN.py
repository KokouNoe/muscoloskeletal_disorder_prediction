import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Supposons que les données sont dans deux variables : X (les caractéristiques) et y (la cible)
# X est de taille (1721, 15) et y est de taille (1721,)
# Les données sont fictives pour cet exemple

# importer le dataset

df=pd.read_excel('TMC_encode.xlsx')

y =df['Résultat_dépistage']

X=df.drop(['Résultat_dépistage'],axis=1)

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construction du modèle DNN
model = Sequential()

# Ajout des couches du réseau
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Dropout pour réduire le surapprentissage
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))

# Couche de sortie avec activation sigmoid pour la classification binaire
model.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Évaluation du modèle sur l'ensemble de test
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Affichage des résultats
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Vous pouvez aussi visualiser l'évolution de l'accuracy et de la perte (loss) au cours de l'entraînement
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


from sklearn.model_selection import KFold

# Définition du nombre de folds pour la validation croisée
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Pour stocker les scores d'accuracy pour chaque fold
accuracy_scores = []

# Fonction pour créer un modèle DNN
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_scaled.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Boucle de validation croisée
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Création du modèle
    model = create_model()
    
    # Entraînement du modèle
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Évaluation du modèle
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Calcul de l'accuracy pour ce fold
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)
    print(f"Fold Accuracy: {acc}")

# Affichage des résultats de la validation croisée
print(f"Validation Croisée Accuracy Moyenne: {np.mean(accuracy_scores)}")
print(f"Validation Croisée Accuracy Écart-Type: {np.std(accuracy_scores)}")
