import cv2
import numpy as np
import pyautogui
import serial
import time
import keyboard
from ultralytics import YOLO
import mss

sct = mss.mss()
cv2.setUseOptimized(True)

# Initialisation YOLO
model = YOLO("best_fort.pt").to("cuda")

# Paramètres YOLO
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Initialisation Arduino
ser = serial.Serial('COM10', 9500, timeout=0)
time.sleep(2)

# Résolution écran et détection
screen_width, screen_height = pyautogui.size()
frame_width, frame_height = 600, 600

# Résolution du jeu
GAME_WIDTH = 1920
GAME_HEIGHT = 1080
SCREEN_TO_GAME_X = GAME_WIDTH / frame_width
SCREEN_TO_GAME_Y = GAME_HEIGHT / frame_height

# Paramètres d'ajustement
ALPHA = 0.4
DEAD_ZONE = 5
SCALE_FACTOR = 3
OFFSET_X = 10
OFFSET_Y = 150

# Variables de suivi
last_x, last_y = screen_width // 2, screen_height // 2
detection_active = True
active_target = None
tracking_loss_timer = 0

# Zone d'exclusion (x_min, y_min, x_max, y_max)
EXCLUDED_REGION = (100, 250, 200, 400) # fortnite essai 

# Fonction pour capturer l'écran
def get_screen():
    monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
    return frame

# Fonction pour envoyer des commandes à l'Arduino
def send_command(x, y):
    global last_x, last_y

    # Conversion des coordonnées de YOLO (repère 400x400) vers l'écran du FPS (repère 1920x1080)
    real_x = int(x * (GAME_WIDTH / frame_width))
    real_y = int(y * (GAME_HEIGHT / frame_height))

    # Vérification pour éviter l'envoi de commandes inutiles dans la zone morte
    if abs(real_x - GAME_WIDTH // 2) < DEAD_ZONE and abs(real_y - GAME_HEIGHT // 2) < DEAD_ZONE:
        return

    # Calcul des déplacements relatifs
    move_x = real_x - GAME_WIDTH // 2
    move_y = real_y - GAME_HEIGHT // 2

    # Vérification que le déplacement est significatif
    if abs(move_x) > 1 or abs(move_y) > 1:
        command = f"{move_x},{move_y}\n"
        ser.write(command.encode("utf-8"))
        print(f"Envoyé à Arduino: {command}")
        last_x, last_y = real_x, real_y

        # Retourner les données pour les afficher avec OpenCV
        return move_x, move_y

    return None, None  # Pas de mouvement envoyé

# Fonction pour vérifier si un objet est dans la zone d'exclusion
def is_within_exclusion_zone(x1, y1, x2, y2):
    x_min, y_min, x_max, y_max = EXCLUDED_REGION
    return (x1 < x_max and x2 > x_min and y1 < y_max and y2 > y_min)

# Fonction pour activer/désactiver la détection
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    print("Détection activée" if detection_active else "Détection désactivée")

keyboard.add_hotkey("ctrl", toggle_detection)

# Sélection de la cible
def select_target(boxes):
    global active_target, tracking_loss_timer
    if active_target:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            if abs(center_x - active_target[0]) < 50 and abs(center_y - active_target[1]) < 50:
                active_target = (center_x, center_y)
                tracking_loss_timer = 0
                return active_target
    tracking_loss_timer += 1
    if tracking_loss_timer > 15:
        active_target = None
    closest_target = None
    closest_distance = float("inf")
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        distance = ((center_x - frame_width // 2) ** 2 + (center_y - frame_height // 2) ** 2) ** 0.5
        if distance < closest_distance:
            closest_distance = distance
            closest_target = (center_x, center_y)
    active_target = closest_target
    return active_target

# Boucle principale
while True:
    if detection_active:
        frame = get_screen()
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, classes=[0], half=True, device="cuda")
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if not is_within_exclusion_zone(x1, y1, x2, y2):
                    target = select_target(result.boxes)
                    if target:
                        send_command(*target)
                        cv2.circle(frame, target, 10, (0, 255, 0), -1)
                        cv2.putText(frame, f"Cible: {target}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        x_min, y_min, x_max, y_max = EXCLUDED_REGION
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
ser.close()
cv2.destroyAllWindows()
