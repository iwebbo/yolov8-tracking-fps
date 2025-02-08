import cv2
import numpy as np
import mss

# Capture d'écran du jeu
sct = mss.mss()
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Ajuste selon ton écran
screenshot = sct.grab(monitor)
frame = np.array(screenshot)
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# Variables globales pour la sélection
drawing = False  # Indique si on est en train de dessiner
x_start, y_start, x_end, y_end = -1, -1, -1, -1

# Fonction de gestion de la souris
def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Début du tracé
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # Suivi du tracé
        if drawing:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (x_start, y_start), (x, y), (0, 0, 255), 2)
            cv2.imshow("Définis la zone d'exclusion", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:  # Fin du tracé
        drawing = False
        x_end, y_end = x, y
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.imshow("Définis la zone d'exclusion", frame)

# Affichage et interaction
cv2.imshow("Définis la zone d'exclusion", frame)
cv2.setMouseCallback("Définis la zone d'exclusion", draw_rectangle)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quitter avec 'q'
        break

cv2.destroyAllWindows()

# Affichage des coordonnées finales
if x_start != -1 and y_start != -1 and x_end != -1 and y_end != -1:
    EXCLUDED_REGION = (x_start, y_start, x_end, y_end)
    print(f"Zone d'exclusion définie: {EXCLUDED_REGION}")
else:
    print("Aucune zone d'exclusion définie.")
