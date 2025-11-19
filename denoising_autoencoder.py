import numpy as np
import os
import shutil
import random  # Nieuw: Nodig voor het selecteren van een willekeurige afbeelding
import cv2  # Nieuw: Nodig voor het inlezen van afbeeldingen
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

# --- Constanten ---
MODEL_PATH = "denoising_autoencoder.keras"
IMAGE_SHAPE = (28, 28)


# --- NIEUWE MODULAIRE FUNCTIE VOOR AFBEELDINGEN ---

def load_random_image_from_folder(folder_path):
    """
    Laadt een willekeurige afbeelding uit de gespecificeerde map,
    zet deze om naar grijswaarden, resize naar 28x28 en normaliseert naar [0, 1].

    Args:
        folder_path (str): Het pad naar de map (bijv. "undistorted_images").

    Returns:
        np.array: De genormaliseerde, 28x28 afbeelding in grijswaarden, of None bij een fout.
    """
    # 1. Haal alle bestanden uit de map
    if not os.path.isdir(folder_path):
        print(f"Fout: Map '{folder_path}' niet gevonden.")
        return None

    # Filter op veelvoorkomende afbeeldings-extensies
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print(f"Fout: Geen afbeeldingen gevonden in '{folder_path}'.")
        return None

    # 2. Selecteer een willekeurige afbeelding
    random_filename = random.choice(image_files)
    file_path = os.path.join(folder_path, random_filename)

    print(f"🖼Laad willekeurige afbeelding: {file_path}")

    # 3. Laad de afbeelding met OpenCV
    # De vlag cv2.IMREAD_GRAYSCALE zorgt voor een 2D array (H, W)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Fout bij het inlezen van de afbeelding: {file_path}")
        return None

    # 4. Resize naar de gewenste modelgrootte (28x28 voor dit MNIST-model)
    img_resized = cv2.resize(img, IMAGE_SHAPE, interpolation=cv2.INTER_AREA)

    # 5. Normaliseren naar [0, 1] float32
    img_normalized = img.astype('float32') / 255.0

    return img_normalized


# ----------------------------------------------------------------------
# (De overige functies blijven ongewijzigd, alleen de imports zijn bovenaan aangepast.)
# ----------------------------------------------------------------------

# --- Visualisatie Functie ---

def plot_images(x, num_images=10):
    """
    Toont een reeks van afbeeldingen in een raster.

    Args:
        x (np.array): De afbeeldingen (afgevlakt, 784 elementen per rij) om weer te geven.
        num_images (int): Het aantal afbeeldingen om te plotten.
    """
    plt.figure(figsize=(20, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(x[i].reshape(IMAGE_SHAPE), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return


# --- Data Voorbereidings Functies ---

def load_and_prepare_data():
    """
    Laadt de MNIST dataset, normaliseert deze en maakt de afbeeldingen plat.
    """
    print("Data laden en voorbereiden (MNIST)...")
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    X_test_raw = x_test.copy()

    x_train_flat = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test_flat = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print(f"Train set shape: {x_train_flat.shape}")
    return x_train_flat, x_test_flat, X_test_raw


def add_noise(data, noise_factor=0.5):
    """
    Voegt willekeurige (Gaussiaanse) ruis toe aan de NumPy array.
    """
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    noisy_data = data + noise
    noisy_data = np.clip(noisy_data, 0., 1.)
    return noisy_data


# --- Model Functies ---

def create_denoising_autoencoder():
    """Definieert het architectuurmodel van de denoising autoencoder."""
    input_image = Input(shape=(np.prod(IMAGE_SHAPE),))
    encoded = Dense(64, activation='relu')(input_image)
    decoded = Dense(np.prod(IMAGE_SHAPE), activation='sigmoid')(encoded)
    autoencoder = Model(input_image, decoded)
    return autoencoder


def train_autoencoder(X_train_noisy, X_train, model_filepath=MODEL_PATH):
    """Compileert, traint en slaat de autoencoder op als een .keras bestand."""
    autoencoder = create_denoising_autoencoder()
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    print(f"\nStart training en opslag in '{model_filepath}'...")
    autoencoder.fit(
        X_train_noisy, X_train,
        epochs=100, batch_size=512, validation_split=0.2, verbose=False,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5),
            LambdaCallback(on_epoch_end=lambda e, l: print('{:.4f}'.format(l['val_loss']), end=' _ '))
        ]
    )
    print('\nTraining is voltooid!')
    autoencoder.save(model_filepath)
    print(f"Model succesvol opgeslagen als: {model_filepath}")
    return autoencoder


def denoise_batch(autoencoder_or_path, noisy_images_flat):
    """Voert de denoising-voorspelling uit op een batch afgevlakte afbeeldingen."""
    if isinstance(autoencoder_or_path, str):
        try:
            model = load_model(autoencoder_or_path)
        except Exception as e:
            raise IOError(f"Fout bij het laden van het model: {e}")
    else:
        model = autoencoder_or_path

    print("🔮 Voorspellingen genereren (batch)...")
    denoised_images = model.predict(noisy_images_flat, verbose=0)
    return denoised_images


def denoise_single_image(autoencoder_or_path, image_array):
    """
    Voert de denoising uit op een enkele, niet-afgevlakte afbeelding.

    Args:
        autoencoder_or_path (Model of str): Het model of het pad naar het .keras bestand.
        image_array (np.array): Een enkele, niet-afgevlakte afbeelding (bijv. shape (28, 28)), genormaliseerd naar [0, 1].

    Returns:
        np.array: De denoised afbeelding in de originele vorm (bijv. shape (28, 28)).
    """
    if isinstance(autoencoder_or_path, str):
        try:
            model = load_model(autoencoder_or_path)
        except Exception as e:
            raise IOError(f"Fout bij het laden van het model: {e}")
    else:
        model = autoencoder_or_path

    flat_input = image_array.reshape(1, np.prod(image_array.shape))
    flat_output = model.predict(flat_input, verbose=0)
    denoised_image = flat_output.reshape(image_array.shape)

    return denoised_image


# --- Volledige Pipeline Functie ---

def full_denoising_pipeline():
    """
    Voert de volledige pipeline uit voor het testen en opslaan van de autoencoder.
    """
    print("--- Start Denoising Autoencoder Pipeline ---")

    X_train_flat, X_test_flat, X_test_raw = load_and_prepare_data()

    X_train_noisy = add_noise(X_train_flat, noise_factor=0.5)
    X_test_noisy_flat = add_noise(X_test_flat, noise_factor=0.5)

    autoencoder = train_autoencoder(X_train_noisy, X_train_flat, model_filepath=MODEL_PATH)

    denoised_images_flat = denoise_batch(autoencoder, X_test_noisy_flat)

    print("\n--- Resultaten Visualisatie (eerste 10 testafbeeldingen) ---")
    plot_images(X_test_flat, num_images=10)
    print("\nAfbeeldingen met Ruis (Input voor Denoising):")
    plot_images(X_test_noisy_flat, num_images=10)
    print("\nAfbeeldingen na Denoising (Output van Model):")
    plot_images(denoised_images_flat, num_images=10)

    print("--- Einde Pipeline ---")

    return autoencoder, X_test_raw, X_test_noisy_flat, denoised_images_flat


# --- Main-blok om de pipeline te draaien bij directe uitvoering ---

if __name__ == '__main__':
    # 1. Draai de volledige pipeline
    trained_model, X_clean_raw, X_noisy_flat, X_denoised_flat = full_denoising_pipeline()
    del trained_model

    print("\n\n*** Modulaire Test: Gebruik van de load_random_image_from_folder functie ***")

    # 2. Test of de nieuwe functie werkt (opmerking: de map 'undistorted_images' moet lokaal bestaan)
    TEST_FOLDER = "undistorted_images"

    # Maak de testmap aan als deze niet bestaat, om een FileNotFoundError te voorkomen
    if not os.path.isdir(TEST_FOLDER):
        print(
            f"Map '{TEST_FOLDER}' niet gevonden. Maak een lege map aan voor testdoeleinden, maar de functie zal falen totdat u afbeeldingen toevoegt.")
        os.makedirs(TEST_FOLDER, exist_ok=True)

    single_image_from_file = load_random_image_from_folder(TEST_FOLDER)

    if single_image_from_file is not None:

        # 3. Voeg ruis toe
        single_noisy_image = add_noise(single_image_from_file, noise_factor=0.5)

        # 4. Denoisen met het getrainde model
        try:
            single_denoised_image = denoise_single_image(MODEL_PATH, single_noisy_image)

            print(f"Afbeelding van shape {single_denoised_image.shape} succesvol denoised met ingeladen model.")

            # Visualiseer de resultaten
            plt.figure(figsize=(6, 2))

            plt.subplot(1, 3, 1)
            plt.imshow(single_image_from_file, cmap='binary')
            plt.title("Schoon (File)")
            plt.xticks([]);
            plt.yticks([])

            plt.subplot(1, 3, 2)
            plt.imshow(single_noisy_image, cmap='binary')
            plt.title("Ruis (Input)")
            plt.xticks([]);
            plt.yticks([])

            plt.subplot(1, 3, 3)
            plt.imshow(single_denoised_image, cmap='binary')
            plt.title("Gedenoised (Output)")
            plt.xticks([]);
            plt.yticks([])

            plt.show()

        except IOError as e:
            print(f"Kan de denoising test niet uitvoeren: {e}")
    else:
        print(
            "\nKan de modulaire test met lokale bestanden niet uitvoeren omdat er geen geschikte afbeelding is geladen.")