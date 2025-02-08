import os
import re
import unicodedata

# 📂 Dossier contenant les images
INPUT_DIR = "dataset/labels/"

# 📌 Fonction de nettoyage du nom de fichier
def clean_filename(filename):
    name, ext = os.path.splitext(filename)  # Séparer le nom et l'extension
    
    # Convertir en minuscule
    name = name.lower()
    
    # Supprimer les accents
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    
    # Remplacer les espaces par des underscores
    name = name.replace(" ", "_")
    
    # Supprimer tous les caractères spéciaux sauf "_", ".", "-"
    name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)
    
    return name + ext  # Rassembler le nom et l'extension

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
