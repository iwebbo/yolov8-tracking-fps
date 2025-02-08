import os
import glob
import cv2
import numpy as np
import albumentations as A

# === 📌 CONFIGURATION ===
INPUT_IMAGES_DIR = "dataset/images/"
INPUT_LABELS_DIR = "dataset/labels/"
OUTPUT_IMAGES_DIR = "dataset_augmented/images/"
OUTPUT_LABELS_DIR = "dataset_augmented/labels/"
AUGMENTATIONS_PER_IMAGE = 15  # Nombre de variations par image

# Créer les dossiers de sortie si besoin
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# === 📌 FONCTION DE TRANSFORMATION ===
transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.ISONoise(p=0.3),
    A.RandomScale(scale_limit=0.1, p=0.4),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

# === 📌 CHARGER & APPLIQUER LES AUGMENTATIONS ===
def load_yolo_annotations(label_path):
    """ Charge les annotations YOLO depuis un fichier texte """
    with open(label_path, "r") as file:
        lines = file.readlines()
    bboxes, class_labels = [], []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_labels.append(int(parts[0]))  # Classe YOLO
        bbox = list(map(float, parts[1:5]))  # x, y, w, h
        bboxes.append(bbox)
    
    return bboxes, class_labels

def save_yolo_annotations(output_path, bboxes, class_labels):
    """ Sauvegarde les annotations YOLO après augmentation """
    with open(output_path, "w") as file:
        for cls, bbox in zip(class_labels, bboxes):
            file.write(f"{cls} {' '.join(map(str, bbox))}\n")

# === 📌 LANCEMENT DE L'AUGMENTATION ===
image_paths = glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.png"))

for image_path in image_paths:
    filename = os.path.basename(image_path).split(".")[0]
    label_path = os.path.join(INPUT_LABELS_DIR, filename + ".txt")

    if not os.path.exists(label_path):
        print(f"❌ Aucune annotation trouvée pour {filename}, passage.")
        continue  

    # Charger image & annotations
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Erreur de chargement de l'image {filename}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = load_yolo_annotations(label_path)

    if not bboxes:
        print(f"⚠️ Aucune bbox valide pour {filename}, passage.")
        continue

    # Générer des augmentations
    for i in range(AUGMENTATIONS_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        aug_image = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        if not aug_bboxes:
            print(f"⚠️ Aucune bbox après augmentation pour {filename}_aug_{i}, passage.")
            continue
        
        # Sauvegarde des images et labels
        aug_image_filename = f"{filename}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, aug_image_filename), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        aug_label_filename = f"{filename}_aug_{i}.txt"
        save_yolo_annotations(os.path.join(OUTPUT_LABELS_DIR, aug_label_filename), aug_bboxes, aug_labels)

    print(f"✅ Augmentations terminées pour {filename}")

print("🚀 Data Augmentation terminée avec succès !")
