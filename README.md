# README : Custom Training avec YOLOv8 et Conversion ONNX

Ce document explique comment effectuer l'entraînement d'un modèle YOLOv8 sur un dataset personnalisé pour la détection d'objets, ainsi que la conversion du modèle final au format ONNX pour une intégration dans un script.

## Prérequis
- **Python 3.8+**
- **PyTorch**
- **Ultralytics YOLOv8** (installer via pip : `pip install ultralytics`)
- **onnx** et **onnxruntime** (installer via pip : `pip install onnx onnxruntime`)
- Un dataset personnalisé annoté dans le format YOLO (à l’aide d’outils comme [LabelImg](https://github.com/heartexlabs/labelImg) ou [Roboflow](https://roboflow.com/)).

## Étapes de Training avec un Dataset Personnalisé

### 1. Organisation du Dataset
Le dataset doit être structuré comme suit :
```
custom_dataset/
|-- train/images/  # Images pour l'entraînement
|-- train/labels/  # Labels correspondants au format YOLO (.txt)
|-- val/images/    # Images pour la validation
|-- val/labels/    # Labels correspondants pour la validation
```
Chaque fichier `.txt` doit avoir les annotations des objets au format YOLO :
```
<class_id> <x_center> <y_center> <width> <height>
```
- `class_id` : ID de la classe
- `x_center`, `y_center`, `width`, `height` : coordonnées normalisées (entre 0 et 1).

### 2. Configuration d’un fichier YAML
Créez un fichier `custom_dataset.yaml` contenant :
```yaml
task: detect
train: path/to/custom_dataset/train
val: path/to/custom_dataset/val

names:
  0: Enemy
  1: Weapon
```
Remplacez les chemins et les noms des classes par ceux de votre dataset.

### 3. Lancement de l’entraînement
Utilisez le package `ultralytics` pour entraîner YOLOv8 :
```python
from ultralytics import YOLO

# Charger le modèle pré-entraîné
model = YOLO('yolov8n.pt')

# Entraîner le modèle sur le dataset custom
model.train(
    data='custom_dataset.yaml',
    epochs=50,       # Nombre d'époques
    imgsz=640,       # Dimension des images
    batch=16,        # Taille des batchs
    name='custom_model'
)
```

Le modèle entraîné sera sauvegardé dans le dossier `runs/detect/custom_model/`.

## Conversion au Format ONNX

Après l’entraînement, vous pouvez convertir le modèle YOLOv8 au format ONNX pour l’intégrer dans votre script :

### 1. Conversion
Utilisez le code suivant pour exporter le modèle :
```python
# Exporter le modèle au format ONNX
model.export(format='onnx')
```
Le fichier ONNX sera généré dans le dossier `runs/detect/export/`.

### 2. Validation du Modèle ONNX
Vérifiez le modèle ONNX pour garantir sa compatibilité :
```python
import onnx
import onnxruntime as ort

# Charger le modèle ONNX
onnx_model = 'runs/detect/custom_model/weights/best.onnx'
session = ort.InferenceSession(onnx_model)

# Vérification des entrées/sorties
print("Inputs:", session.get_inputs())
print("Outputs:", session.get_outputs())
```

## Intégration dans le Script de Détection

### Exemple de Chargement et Inference avec ONNX
```python
import cv2
import numpy as np
import onnxruntime as ort

# Charger le modèle ONNX
onnx_model = 'runs/detect/custom_model/weights/best.onnx'
session = ort.InferenceSession(onnx_model)

# Prétraitement de l'image
def preprocess(image, input_size):
    img = cv2.resize(image, (input_size, input_size))
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = img / 255.0  # Normalisation
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# Inference
def detect(image):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_size = session.get_inputs()[0].shape[-1]  # Taille des entrées

    img = preprocess(image, input_size)
    preds = session.run([output_name], {input_name: img})[0]
    return preds
```

Ce script charge le modèle ONNX, prétraite une image et exécute l'inférence pour obtenir les prédictions.

## Notes
- Vous pouvez optimiser le modèle ONNX pour des performances accrues avec des outils comme [ONNX Runtime](https://onnxruntime.ai/).
- Assurez-vous que vos annotations sont précises pour obtenir un modèle entraîné efficace.
- Testez le modèle dans des conditions proches de votre scénario réel pour valider les performances.

## Conclusion
Avec ces étapes, vous pouvez entraîner un modèle YOLOv8 sur un dataset personnalisé et le convertir au format ONNX pour une intégration facile dans vos projets.

