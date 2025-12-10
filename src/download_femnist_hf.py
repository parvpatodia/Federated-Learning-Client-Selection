import os
import numpy as np

print("Installing Hugging Face datasets library")
os.system("pip install datasets pillow -q")

print("\nDownloading REAL FEMNIST from Hugging Face")
print("="*80)

from datasets import load_dataset

try:
    print("Loading FEMNIST dataset (3,550 users, 805k samples)")
    dataset = load_dataset("flwrlabs/femnist")

    print(f" Downloaded successfully!")
    print(f"  Dataset shape: {dataset}")

    # Save info
    os.makedirs("data/femnist_hf", exist_ok=True)

    with open("data/femnist_hf/info.txt", "w") as f:
        f.write(f"FEMNIST Dataset from Hugging Face\n")
        f.write(f"Users: 3,550\n")
        f.write(f"Total samples: 805,263\n")
        f.write(f"Image size: 28x28 grayscale\n")
        f.write(f"Classes: 62 (10 digits + 52 letters)\n")
        f.write(f"Dataset: {dataset}\n")

    print("\n" + "="*80)
    print(" REAL FEMNIST ready to use!")
    print("="*80)

except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTrying alternative download method")