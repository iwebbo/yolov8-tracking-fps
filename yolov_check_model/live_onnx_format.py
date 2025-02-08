import cv2
import numpy as np
import pyautogui
import serial
import time
import keyboard
import mss
import onnxruntime as ort

sct = mss.mss()

# Charger le modèle ONNX
onnx_model_path = "best.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

# Vérification du support GPU
print("Fournisseurs disponibles :", ort.get_available_providers())

# Paramètres de détection
CONFIDENCE_THRESHOLD = 0.5

# Initialisation Arduino
ser = serial.Serial('COM10', 115200, timeout=0)
time.sleep(2)

# Taille d'entrée du modèle (doit correspondre au modèle)
frame_width, frame_height = 416, 416

# Résolution du jeu
GAME_WIDTH = 1920
GAME_HEIGHT = 1080
SCREEN_TO_GAME_X = GAME_WIDTH / frame_width
SCREEN_TO_GAME_Y = GAME_HEIGHT / frame_height

# Zone d'exclusion
EXCLUDED_REGION = (100, 250, 200, 400)

# Variables de suivi
detection_active = True

def get_screen():
    monitor = {"top": 0, "left": 0, "width": pyautogui.size()[0], "height": pyautogui.size()[1]}
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
    return frame


def send_command(x, y):
    real_x = int(x * SCREEN_TO_GAME_X)
    real_y = int(y * SCREEN_TO_GAME_Y)

    if abs(real_x - GAME_WIDTH // 2) < 80 and abs(real_y - GAME_HEIGHT // 2) < 80:
        return

    move_x = real_x - GAME_WIDTH // 2
    move_y = real_y - GAME_HEIGHT // 2

    if abs(move_x) > 1 or abs(move_y) > 1:
        command = f"{move_x},{move_y}\n"
        ser.write(command.encode("utf-8"))
        print(f"Envoyé à Arduino: {command}")

def is_within_exclusion_zone(x1, y1, x2, y2):
    x_min, y_min, x_max, y_max = EXCLUDED_REGION
    return (x1 < x_max and x2 > x_min and y1 < y_max and y2 > y_min)

def toggle_detection():
    global detection_active
    detection_active = not detection_active
    print("Détection activée" if detection_active else "Détection désactivée")

keyboard.add_hotkey("ctrl", toggle_detection)

while True:
    if detection_active:
        frame = get_screen()

        # Prétraitement de l'image pour ONNX
        blob = cv2.resize(frame, (frame_width, frame_height))
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Nom de l'entrée : {input_name}, Dimensions attendues : {input_shape}")
        outputs = session.run(None, {input_name: blob})

        # La première sortie contient les détections
        detections = outputs[0]  # Vérifie si l'index de sortie est correct
        print(f"Détections : {detections.shape}")  # Debug pour vérifier la forme

        if len(detections) > 0:  # Vérifie si des détections existent
            for detection in detections:
                if detection.ndim == 1 and len(detection) >= 5:
                    x1, y1, x2, y2 = (detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])).astype(int)
                    confidence = float(detection[4])

                    # Corriger les coordonnées hors cadre
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)

                    if confidence > CONFIDENCE_THRESHOLD and not is_within_exclusion_zone(x1, y1, x2, y2):
                        # Calculer le centre et envoyer les commandes
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        send_command(center_x, center_y)

                        # Dessiner un rectangle et des informations sur l'image
                        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                        cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Dessiner la zone d'exclusion
        x_min, y_min, x_max, y_max = EXCLUDED_REGION
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    cv2.imshow("YOLOv8 ONNX Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cv2.destroyAllWindows()
