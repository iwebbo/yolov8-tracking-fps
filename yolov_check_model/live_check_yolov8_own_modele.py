import cv2
import numpy as np
import pyautogui
import torch
from ultralytics import YOLO

# Définir l'appareil à utiliser (GPU si disponible, sinon CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilisation de {device}")

# Charger le modèle YOLOv8 personnalisé
model = YOLO("best_fort.pt").to(device)
print("Modèle YOLOv8 chargé avec succès et prêt pour l'inférence !")

# Paramètres de détection
CONFIDENCE_THRESHOLD = 0.5  # Seuil de confiance pour les détections
IOU_THRESHOLD = 0.45  # Seuil de l'Intersection over Union pour le NMS

# Obtenir la taille de l'écran principal
screen_width, screen_height = pyautogui.size()

# Taille de redimensionnement pour l'affichage
frame_width, frame_height = 640, 480

def get_screen():
    # Capturer une capture d'écran de l'écran principal
    screenshot = pyautogui.screenshot(region=(0, 0, screen_width, screen_height))
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

while True:
    # Obtenir la capture d'écran
    frame = get_screen()

    # Redimensionner le cadre pour l'affichage
    frame_resized = cv2.resize(frame, (frame_width, frame_height))

    # Effectuer la détection d'objets
    results = model(frame_resized, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

    # Parcourir les résultats et dessiner les boîtes englobantes
    for result in results:
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                # Dessiner la boîte englobante et l'étiquette
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher le cadre avec les détections
    cv2.imshow("Detection", frame_resized)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cv2.destroyAllWindows()
