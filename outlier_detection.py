import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, losses, Model, models

# --- CONFIGURATIE ---
IMG_SIZE = (32, 32)
LATENT_DIM = 64
BATCH_SIZE = 32

# ==========================================
# 1. HULPFUNCTIES (Pre-processing & Metrics)
# ==========================================

def preprocess_image(image_input):
    """
    image_input: pad naar afbeelding (str) OF numpy array
    """

    # CASE 1: path → load file
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Kon afbeelding niet laden: {image_input}")

    # CASE 2: numpy array → convert
    elif isinstance(image_input, np.ndarray):
        # Convert to grayscale if needed
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    else:
        raise TypeError("image_input moet een pad of numpy array zijn.")

    # Resize to model input shape
    img = cv2.resize(img, (32, 32))
    img = img.astype("float32") / 255.0

    return img


def calculate_mse(original, reconstructed):
    """Berekent de Mean Squared Error."""
    return mean_squared_error(original.flatten(), reconstructed.flatten())

# ==========================================
# 2. DE GEVRAAGDE FUNCTIE (INFERENCE)
# ==========================================

def check_image_is_outlier(image_input, model_path="model.keras", threshold_path="threshold.txt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model niet gevonden op: {model_path}")

    model = models.load_model(model_path)

    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Threshold bestand niet gevonden op: {threshold_path}")

    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())

    # image_input can be path OR numpy array
    img = preprocess_image(image_input)

    img_batch = np.expand_dims(img, axis=0)

    reconstruction = model.predict(img_batch, verbose=0)[0]

    mse = calculate_mse(img, reconstruction)

    is_outlier = mse > threshold

    return is_outlier, mse



# ==========================================
# 3. MODEL BOUWEN & TRAINEN (De 'Training Pipeline')
# ==========================================

def build_autoencoder(input_shape=(32, 32), latent_dim=LATENT_DIM):
    """Bouwt de Autoencoder architectuur."""
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Reshape((32, 32, 1))(encoder_input)
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation='tanh')(x)

    # Decoder
    x = layers.Dense(8*8*64, activation='relu')(latent)
    x = layers.Reshape((8, 8, 64))(x)
    x = layers.Conv2DTranspose(64, (3,3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2DTranspose(32, (3,3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    decoder_output = layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')(x)
    decoder_output = layers.Reshape((32, 32))(decoder_output)

    autoencoder = Model(encoder_input, decoder_output, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    return autoencoder

def load_data_from_folder(folder_path):
    """Laadt afbeeldingen uit een map."""
    images = []
    if not os.path.exists(folder_path):
        return np.array([])

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith('.png'):
            try:
                img = preprocess_image(os.path.join(folder_path, img_name))
                images.append(img)
            except:
                pass
    return np.stack(images) if images else np.array([])

def run_training_pipeline(normal_data_path, anomaly_data_path):
    """
    Laadt data, traint model, berekent threshold en slaat alles op.
    """
    print("1. Data laden...")
    normal_imgs = load_data_from_folder(normal_data_path)
    anomaly_imgs = load_data_from_folder(anomaly_data_path) # Alleen nodig voor validatie threshold

    if len(normal_imgs) < 10:
        print("Te weinig data om te trainen.")
        return

    # Splitsen
    x_val_normal = normal_imgs[:5] # 5 apart houden voor validatie
    x_remaining = normal_imgs[5:]
    x_train, x_test = train_test_split(x_remaining, test_size=0.2, random_state=42)

    print(f"2. Model trainen op {len(x_train)} afbeeldingen...")
    model = build_autoencoder()
    model.fit(x_train, x_train, epochs=50, batch_size=BATCH_SIZE, validation_data=(x_test, x_test), verbose=1)

    print("3. Threshold berekenen...")
    # We gebruiken de normale validatie set om de 'normale' error te bepalen
    reconstructions = model.predict(x_val_normal, verbose=0)
    mse_scores = [calculate_mse(x_val_normal[i], reconstructions[i]) for i in range(len(x_val_normal))]

    # Threshold = gemiddelde error + 2x standaardafwijking (of een vaste waarde uit je notebook)
    threshold = np.mean(mse_scores) + (2 * np.std(mse_scores))

    # Zorg dat threshold niet te laag is (veiligheidsmarge)
    threshold = max(threshold, 0.005)

    print(f"Berekende Threshold: {threshold}")

    # Opslaan
    model.save("model.keras")
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))

    print("4. Model en threshold opgeslagen.")

# ==========================================
# 4. HOOFD PROGRAMMA
# ==========================================
if __name__ == "__main__":
    # STAP 1: TRAINEN (Zet dit op False als je al een model hebt en alleen wilt checken)
    MOET_NOG_TRAINEN = True

    PATH_NORMAL = "image_data/image_data"
    PATH_ANOMALY = "Nieuwe plaatjes"

    if MOET_NOG_TRAINEN:
        if os.path.exists(PATH_NORMAL):
            run_training_pipeline(PATH_NORMAL, PATH_ANOMALY)
        else:
            print("Geen data gevonden om te trainen. Zorg dat de mappen bestaan.")

    # STAP 2: EEN FOTO CHECKEN MET DE NIEUWE FUNCTIE
    # Dit simuleert het gebruik in productie
    print("\n--- Start detectie op losse foto ---")

    # Pak een willekeurige testfoto (bijv. de eerste uit de anomalie map)
    if os.path.exists(PATH_ANOMALY) and len(os.listdir(PATH_ANOMALY)) > 0:
        test_foto = os.path.join(PATH_ANOMALY, os.listdir(PATH_ANOMALY)[0])

        # AANROEP VAN DE GEVRAAGDE FUNCTIE
        is_outlier, mse = check_image_is_outlier(
            image_path=test_foto,
            model_path="model.keras",
            threshold_path="threshold.txt"
        )

        if is_outlier is not None:
            print(f"Functie output -> Is Outlier: {is_outlier}, MSE: {mse}")