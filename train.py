import json
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img

# Charger les annotations
with open('data/flowIA-annotations.json', 'r') as f:
    annotations = json.load(f)

# Créer les ensembles de données
labels_map = {"oui": 0, "non": 1, "autre": 2}
images = []
labels = []

for annotation in annotations["annotations"]:
    file_path = os.path.join("data/images", annotation["fileName"])
    if os.path.exists(file_path):
        try:
            img = load_img(file_path, target_size=(128, 128))
            images.append(tf.keras.preprocessing.image.img_to_array(img) / 255.0)
            labels.append(labels_map[annotation["annotation"]["label"]])
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {file_path}: {e}")
    else:
        print(f"Image introuvable : {file_path}")

# Vérification des données
if len(images) == 0:
    raise ValueError("Aucune image valide trouvée. Vérifiez vos annotations et vos chemins d'accès.")

# Convertir en tableaux numpy
images = np.array(images)
labels = np.array(labels)

# Séparation des ensembles de données
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Modèle léger
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)
model.save("model.h5")
