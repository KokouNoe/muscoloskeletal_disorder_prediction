# Installation et chargement des packages nécessaires
install.packages("keras")
library(keras)

# Si vous n'avez pas encore installé TensorFlow, exécutez la commande suivante:
install_keras()

# Supposons que les données sont dans deux variables : X (les caractéristiques) et y (la cible)
# Création de données fictives
set.seed(42)
X <- matrix(runif(1721 * 15), nrow = 1721, ncol = 15)
y <- sample(0:1, 1721, replace = TRUE)

# Normalisation des données
X <- scale(X)

# Division des données en ensembles d'entraînement et de test
set.seed(42)
train_indices <- sample(1:nrow(X), size = 0.8 * nrow(X))
X_train <- X[train_indices, ]
y_train <- y[train_indices]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

# Construction du modèle DNN
model <- keras_model_sequential()

model %>%
  layer_dense(units = 64, activation = 'relu', input_shape = c(ncol(X_train))) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compilation du modèle
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c('accuracy')
)

# Entraînement du modèle
history <- model %>% fit(
  X_train, y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2
)

# Évaluation du modèle sur l'ensemble de test
scores <- model %>% evaluate(X_test, y_test)
cat('Test accuracy:', scores["accuracy"], "\n")

# Prédictions sur l'ensemble de test
y_pred <- model %>% predict_classes(X_test)

# Affichage de la matrice de confusion
table(Predicted = y_pred, Actual = y_test)

# Visualisation de l'évolution de l'accuracy et de la perte
plot(history)

