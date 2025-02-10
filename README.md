# README : yolov8-tracking-fps

###ENGLISH VERSION COMING SOON
##📌 Prerequisites

##🖥 Hardware:

A computer capable of running Python and deep learning models.
An Arduino connected via a serial port (e.g., Arduino Uno).
A USB cable to connect the computer and the Arduino.

##🛠 Software:

Python 3.8+
PyTorch
Ultralytics YOLOv8 (pip install ultralytics)
ONNX and ONNX Runtime (pip install onnx onnxruntime)
A custom dataset annotated in YOLO format (using tools like LabelImg or Roboflow).

##🔧 Dataset Organization

Your dataset should follow this structure:
```
custom_dataset/
|-- train/images/  # Training images
|-- train/labels/  # Corresponding YOLO format labels (.txt)
|-- val/images/    # Validation images
|-- val/labels/    # Corresponding validation labels

Each .txt file should contain annotations in YOLO format:

<class_id> <x_center> <y_center> <width> <height>
```
👉 Find datasets: Roboflow Universe


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

- Ou trouver des dataset custom ?
- https://universe.roboflow.com

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

Au sein de cette function, nous définissons quelle modèle utiliser : 
Merci de regarder la documentation officièlle : https://github.com/ultralytics/ultralytics 

Fonctionnalités
Chargement du Modèle : Choisissez entre un modèle neuf, pré-entraîné ou avec transfert d'apprentissage.
Entraînement : Spécifiez le nombre d'époques et la taille des lots pour l'entraînement.
Validation : Option pour valider le modèle après l'entraînement.
Exportation : Possibilité d'exporter le modèle entraîné dans différents formats.

load (str) : Méthode de chargement du modèle.
'new' : Crée un nouveau modèle à partir du fichier YAML.
'pre' : Charge un modèle pré-entraîné.
'tran' : Charge un modèle à partir du YAML et applique un transfert d'apprentissage.
traindata (str) : Chemin vers le fichier de configuration des données d'entraînement (par défaut : "fortnite.yaml").
epochs (int) : Nombre d'époques pour l'entraînement (par défaut : 50).
batch_size (int) : Taille des lots pour l'entraînement (par défaut : 16).
export (bool) : Indique si le modèle doit être exporté après l'entraînement (par défaut : True).
val (bool) : Indique si une validation doit être effectuée après l'entraînement (par défaut : True).

Utilisez le package `ultralytics` pour entraîner YOLOv8 :
```python
def custom_train(load='pre', traindata="fortnite.yaml", epochs=50, batch_size=16, export=True, val=True):
    # Chargement du modèle
    if load == 'new':
        model = YOLO('yolov8n.yaml')  # Construire un nouveau modèle à partir du YAML
    elif load == 'pre':
        model = YOLO('yolov8n.pt')  # Charger un modèle pré-entraîné
    elif load == 'tran':
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # Construire à partir du YAML et transférer les poids
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

### 2. Validation du Modèle PT
## Installation

1.  Assurez-vous d'avoir Python installé.
2.  Installez les librairies requises :
    ```bash
    pip install torch ultralytics opencv-python
    ```

## Utilisation

1.  Placez votre modèle YOLOv8 entraîné (`best_fort.pt`) et l'image à tester (`capture.jpg`) dans le même répertoire que le script.
2.  Modifiez le script :
    *   Mettez à jour `model_path` et `image_path` si nécessaire.
    *   **Surtout, remplacez `["0", "1"]` dans `class_names` par les noms réels de vos classes.**
3.  Exécutez le script :
    ```bash
    python check_from_capture_pt_file_detect.py
    ```

## Résultat

Une fenêtre affichera l'image avec les objets détectés, leurs boîtes englobantes, leurs noms de classes et leurs confiances.
![capture](https://github.com/user-attachments/assets/29fd22e4-8ae5-478f-8973-0abbbff56de0)
![Capture d'écran 2025-02-02 184733](https://github.com/user-attachments/assets/2d829261-e280-41ce-a99a-d0819b8650b9)
![Capture d'écran 2025-02-01 223906](https://github.com/user-attachments/assets/4b017dfa-542e-42fe-a1dd-b44fc40a6575)

## Notes

*   Assurez-vous que votre modèle YOLOv8 est compatible avec la version de `ultralytics` que vous utilisez.
*   Si vous rencontrez des erreurs, vérifiez les chemins des fichiers et les noms de classes.

## Intégration dans le Script de Détection

### 3. Validation du modèle ONNX
Utilisation
Placez votre modèle ONNX (best_fort.onnx) et l'image à tester (capture.jpg) dans le même répertoire que le script.
Modifiez le script :
Mettez à jour model_path et image_path si nécessaire.
Surtout, remplacez ["0", "1"] dans class_names par les noms réels de vos classes.
Ajustez input_size si nécessaire (doit correspondre à la taille d'entrée du modèle).
Exécutez le script :
Bash

python check_from_capture_onnx_file_detect.py
Résultat
Une fenêtre affichera l'image avec les objets détectés, leurs boîtes englobantes, leurs noms de classes et leurs confiances.

Notes
Assurez-vous que votre modèle ONNX est compatible avec ONNX Runtime.
Si vous rencontrez des erreurs, vérifiez les chemins des fichiers, les noms de classes et la taille d'entrée du modèle.
Ce script suppose que la sortie du modèle ONNX est un tableau numpy contenant les détections au format attendu (voir la fonction postprocess_detections). Ce format est typique des modèles YOLO, mais peut varier. Si votre modèle a une sortie différente, vous devrez adapter la fonction postprocess_detections.

**Changements importants et explications :**

*   **Gestion des erreurs d'image:** Ajout d'une gestion des erreurs dans `preprocess_image` pour lever une exception si l'image ne peut pas être lue.
*   **Image originale pour le dessin:** La fonction `preprocess_image` retourne maintenant l'image originale *et* l'image prétraitée. L'image originale est utilisée pour dessiner les boîtes, car elle conserve la résolution originale.
*   **Inférence ONNX:** L'exécution de l'inférence ONNX se fait avec `session.run()`.  Il est *crucial* de récupérer la sortie du modèle correctement. J'ai ajouté `[0]` à la fin pour extraire le résultat du tableau retourné par `session.run()`.  **Vérifiez le format de sortie de votre modèle ONNX**.  Ce code suppose qu'il retourne un seul tenseur contenant les détections.  Si ce n'est pas le cas, vous devrez adapter cette ligne.
*   **Transposition des détections:** La boucle dans `postprocess_detections` utilise `.T` (transposition) pour itérer sur les détections de manière plus intuitive.
*   **Conversion en entiers:** La conversion des coordonnées de boîtes en entiers est faite dans `draw_boxes`, juste avant de dessiner, pour plus de sécurité.
*   **Commentaires:** Ajout de commentaires plus détaillés pour expliquer chaque étape.

## 4 Run & Test
Utilisation
Placez votre modèle YOLOv8 entraîné (best_fort.pt) dans le même répertoire que le script.
Exécutez le script :
Bash

python live_check_yolov8.py

## Paramètres:
CONFIDENCE_THRESHOLD : Seuil de confiance pour les détections (par défaut : 0.5).
IOU_THRESHOLD : Seuil de l'Intersection over Union pour le NMS (par défaut : 0.45).
Vous pouvez ajuster ces paramètres directement dans le script.

## Fonctionnement:
Le script capture une capture d'écran de votre écran, la redimensionne, effectue une détection d'objets avec votre modèle YOLOv8, puis affiche le résultat avec les boîtes englobantes et les étiquettes.
Appuyez sur la touche 'q' pour quitter la boucle et fermer la fenêtre.

## Notes: 
Assurez-vous que votre modèle YOLOv8 est compatible avec la version de ultralytics que vous utilisez.
Si vous rencontrez des erreurs, vérifiez le chemin du fichier de modèle et assurez-vous que les librairies sont correctement installées.
Ce script utilise l'écran principal. Si vous souhaitez capturer un autre écran, vous devrez ajuster les paramètres de pyautogui.screenshot().
Les performances peuvent varier en fonction de la puissance de votre ordinateur et de la taille de l'écran.

## Améliorations possibles
Ajouter des options pour personnaliser la taille de la fenêtre d'affichage.
Permettre de sélectionner un autre écran à capturer.
Ajouter un compteur de FPS (images par seconde) pour évaluer les performances.
Utiliser un thread ou un processus séparé pour la capture d'écran afin d'améliorer les performances.



## 5 Run with an Arduino to tracking a target
##Matériel
Un Arduino connecté à votre ordinateur via le port série.
Un jeu FPS.

##Utilisation
#Placez votre modèle : Copiez votre fichier de modèle YOLOv8 entraîné (best_fort.pt) dans le même répertoire que ce script.
Connectez l'Arduino : Connectez votre Arduino à votre ordinateur et notez le port COM utilisé (par exemple, COM10).

##Modifiez le script :
Remplacez 'COM10' dans la variable ser par le port COM correct de votre Arduino.
Ajustez les paramètres suivants dans le script en fonction de votre configuration :
CONFIDENCE_THRESHOLD : Seuil de confiance pour la détection.
IOU_THRESHOLD : Seuil IOU pour le NMS.
frame_width, frame_height : Taille de la fenêtre de capture et de traitement.
GAME_WIDTH, GAME_HEIGHT : Résolution du jeu.
DEAD_ZONE : Zone morte autour du centre de l'écran où les mouvements ne sont pas envoyés à l'Arduino.
EXCLUDED_REGION : Zone de l'écran à exclure de la détection (utile pour éviter de cibler des éléments de l'interface du jeu).


##Exécutez le script :
Bash

python Aimbot_Assist_w_Arduino.py
Fonctionnement

Le script capture une portion de l'écran, effectue une détection d'objets avec YOLOv8, sélectionne la cible la plus proche du centre de l'écran et envoie les coordonnées de la cible à l'Arduino. L'Arduino peut ensuite être programmé pour contrôler la souris ou d'autres périphériques d'entrée pour viser dans le jeu.

Ctrl : Active/désactive la détection.
Le script affiche une fenêtre avec le flux vidéo, les détections, la cible sélectionnée (un cercle vert) et la zone d'exclusion (un rectangle rouge).

##Configuration du jeu
Assurez-vous que les paramètres du jeu (résolution, sensibilité de la souris) sont configurés de manière appropriée pour fonctionner avec le script.  Il peut être nécessaire d'ajuster les paramètres du script (en particulier SCALE_FACTOR, OFFSET_X, OFFSET_Y) pour une performance optimale.

##Code Arduino (exemple) 
Voici un exemple de code Arduino qui reçoit les commandes du script Python et simule des mouvements de souris :

```
C++
#include <Mouse.h>

void setup() {
  Serial.begin(9500);
  Mouse.begin();
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    int x = command.substring(0, command.indexOf(',')).toInt();
    int y = command.substring(command.indexOf(',') + 1).toInt();

    Mouse.move(x, y, 0); // Déplacer la souris (x, y, 0)
  }
}

```
##Important : Ce code Arduino est un exemple. Vous devrez peut-être l'adapter en fonction de vos besoins et de la façon dont vous souhaitez contrôler le jeu.

##Limitations
Les performances peuvent varier en fonction de la puissance de l'ordinateur, de la complexité du modèle YOLOv8 et de la résolution de l'écran.
Le suivi de cible peut être perdu si l'objet cible est obstrué ou sort de l'écran.
Ce script est un point de départ. Vous devrez probablement l'affiner pour obtenir les résultats souhaités dans votre jeu spécifique.


## Conclusion
Avec ces étapes, vous pouvez entraîner un modèle YOLOv8 sur un dataset personnalisé et le convertir au format ONNX pour une intégration facile dans vos projets.

### Voici un petit exemple pour un FPS 
https://streamable.com/xf2lst 

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lFrydrUvbJc/0.jpg)](https://www.youtube.com/watch?v=lFrydrUvbJc)

https://youtu.be/lFrydrUvbJc?si=hPbMRpqrRUQb6bWZ

Le script fournit gère également les zones d'exclusions, afin que la detection ne se fasse pas dans cette frame.
Si vous voulez trouver et entrainer d'autres modèles, https://universe.roboflow.com 


