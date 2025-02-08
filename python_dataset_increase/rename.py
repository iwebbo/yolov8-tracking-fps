import os
import re

# 📂 Dossier contenant les images
INPUT_DIR = "dataset/labels/"

# 📌 Fonction de nettoyage du nom de fichier
def clean_filename(filename):
    filename = filename.lower()  # Convertir en minuscule
    filename = filename.replace(" ", "_")  # Remplacer les espaces par des underscores
    filename = re.sub(r'[^\w_.-]', '', filename)  # Supprimer les caractères spéciaux sauf "_", ".", "-"
    return filename

# 📌 Renommer les fichiers
for filename in os.listdir(INPUT_DIR):
    old_path = os.path.join(INPUT_DIR, filename)
    
    if os.path.isfile(old_path):
        new_filename = clean_filename(filename)
        new_path = os.path.join(INPUT_DIR, new_filename)

        if old_path != new_path:  
            os.rename(old_path, new_path)
            print(f"✅ {filename} → {new_filename}")

print("🚀 Tous les fichiers ont été renommés correctement !")