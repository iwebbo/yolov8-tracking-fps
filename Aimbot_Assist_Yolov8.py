import cv2
import torch
import pyautogui
import numpy as np
import serial
import time
from PIL import ImageGrab
from ultralytics import YOLO

# Configurer la communication série avec l'Arduino
port = 'COM10'  # Remplacez par le port COM correct (COMx sur Windows, /dev/ttyUSBx sur Linux)
baudrate = 9600
arduino = serial.Serial(port, baudrate, timeout=1)

# Charger le modèle YOLOv8 (modèle pré-entraîné ou modèle personnalisé .pt)
def load_yolov8_model(model_path='codmobile.pt'):
    # Charge le modèle YOLOv8
    model = YOLO(model_path).cpu()  # Charge un modèle YOLOv8, qu'il soit pré-entraîné ou personnalisé
    return model

# Fonction pour la détection des objets dans l'image
def detect_objects(frame, model):
    # Faire une prédiction avec YOLOv8
    results = model(frame)  # Passer l'image dans le modèle YOLOv8
    predictions = results[0].boxes  # Résultats de prédictions (boîtes englobantes)
    detected_objects = []

    # Afficher les résultats pour vérifier la structure
    # print(results)  # Décommentez cette ligne si vous avez besoin de déboguer la structure des résultats

    # Extraire les coordonnées de la boîte englobante et les étiquettes
    for box in predictions:
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()  # Déplacer vers le CPU avant conversion
        confidence = box.conf[0].item()
        cls = int(box.cls[0].item())  # Indice de la classe
        label = model.names[cls]  # Utilisation correcte pour accéder aux noms des classes
        
        detected_objects.append((label, confidence, int(xmin), int(ymin), int(xmax), int(ymax)))
        
        # Dessiner la boîte et le label sur l'image
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame, detected_objects

# Fonction pour déplacer la souris en fonction des coordonnées de l'objet
def move_mouse(detected_objects, screen_width, screen_height):
    if detected_objects:
        # Prendre les coordonnées du premier objet détecté (ou le plus pertinent)
        _, _, xmin, ymin, xmax, ymax = detected_objects[0]
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        # Mapper ces coordonnées à la résolution de l'écran
        mouse_x = int(center_x * screen_width / 1920)
        mouse_y = int(center_y * screen_height / 1080)
        
        # Déplacer la souris via PyAutoGUI
        pyautogui.moveTo(mouse_x, mouse_y)
        print(f"Moved mouse to: ({mouse_x}, {mouse_y})")

        # Envoyer les coordonnées à l'Arduino pour d'autres actions
        #send_coordinates_to_arduino(mouse_x, mouse_y)

# Fonction pour envoyer les coordonnées à l'Arduino via le port COM
def send_coordinates_to_arduino(x, y):
    coordinates = f"{x},{y}\n"
    arduino.write(coordinates.encode())
    print(f"Sent coordinates to Arduino: {coordinates}")

# Fonction principale pour capturer l'écran, détecter les objets et déplacer la souris
def main():
    model = load_yolov8_model('codmobile.pt')  # Charger votre modèle personnalisé YOLOv8
    screen_width, screen_height = pyautogui.size()  # Récupérer la taille de l'écran
    
    while True:
        # Capturer une image de l'écran
        screen = np.array(ImageGrab.grab())  # Capture l'écran
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)  # Convertir l'image au format RGB
        
        # Détection d'objets avec YOLOv8
        frame, detected_objects = detect_objects(frame, model)
        
        # Déplacer la souris en fonction de l'objet détecté
        move_mouse(detected_objects, screen_width, screen_height)

        # Afficher l'image avec les objets détectés
        cv2.imshow("YOLOv8 Object Detection", frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('o'):
            break

    # Fermer la caméra et la communication avec l'Arduino
    cv2.destroyAllWindows()
    arduino.close()

if __name__ == "__main__":
    main()