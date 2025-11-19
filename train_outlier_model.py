import os
import numpy as np
from sklearn.model_selection import train_test_split

# Importeer de nodige functies uit je vorige bestand
# Zorg dat anomaly_detection_v2.py in dezelfde map staat!
try:
    from outlier_detection import load_data_from_folder, build_autoencoder, calculate_mse
except ImportError:
    print("FOUT: Kan 'anomaly_detection_v2.py' niet vinden. Zorg dat dit bestand in dezelfde map staat.")
    exit()

# --- CONFIGURATIE ---
DATA_PATH = "image_data/image_data"  # Pad naar je trainingsdata
OUTPUT_FOLDER = "model_path"  # Map waar het model wordt opgeslagen
EPOCHS = 200  # Aantal epochs zoals gevraagd
BATCH_SIZE = 32


def train_and_save_model():
    print(f"1. Mappen controleren en aanmaken: '{OUTPUT_FOLDER}'...")
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"   Map '{OUTPUT_FOLDER}' aangemaakt.")

    print(f"2. Data inladen uit '{DATA_PATH}'...")
    images = load_data_from_folder(DATA_PATH)

    if len(images) < 20:
        print("   FOUT: Te weinig afbeeldingen gevonden om goed te kunnen trainen.")
        return

    print(f"   {len(images)} afbeeldingen gevonden.")

    # Data splitsen:
    # We houden een deel apart (x_val_threshold) puur om de threshold te berekenen na het trainen
    # De rest splitsen we in train en test voor het fit-proces.
    x_remaining, x_val_threshold = train_test_split(images, test_size=0.1, random_state=42)
    x_train, x_test = train_test_split(x_remaining, test_size=0.2, random_state=42)

    print(f"3. Model bouwen en trainen ({EPOCHS} epochs)...")
    model = build_autoencoder(input_shape=(32, 32))

    # Trainen
    model.fit(
        x_train, x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test, x_test),
        verbose=1  # Zet op 0 om output stil te houden, 1 voor progress bar
    )

    print("4. Threshold berekenen op basis van validatie data...")
    # We gebruiken de aparte validatieset om te kijken wat de 'normale' foutmarge is
    reconstructions = model.predict(x_val_threshold, verbose=0)
    mse_scores = [calculate_mse(x_val_threshold[i], reconstructions[i]) for i in range(len(x_val_threshold))]

    # Bereken threshold: Gemiddelde + 2x Standaardafwijking
    threshold = np.mean(mse_scores) + (2 * np.std(mse_scores))
    # Veiligheidsmarge: zorg dat threshold niet extreem laag is
    threshold = max(threshold, 0.005)

    print(f"   Berekende Threshold: {threshold:.6f}")

    print("5. Opslaan in model_path...")

    # Paden bepalen
    model_save_path = os.path.join(OUTPUT_FOLDER, "autoencoder_200.keras")
    threshold_save_path = os.path.join(OUTPUT_FOLDER, "threshold.txt")

    # Opslaan
    model.save(model_save_path)
    with open(threshold_save_path, "w") as f:
        f.write(str(threshold))

    print("-" * 30)
    print(f"KLAAR! Model opgeslagen in: {model_save_path}")
    print(f"Threshold opgeslagen in:    {threshold_save_path}")
    print("-" * 30)


if __name__ == "__main__":
    train_and_save_model()