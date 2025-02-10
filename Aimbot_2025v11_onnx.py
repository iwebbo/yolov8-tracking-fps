import cv2
import numpy as np
import onnxruntime as ort
import mss
import pyautogui

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (left, top)

# Paramètres d'ajustement
SPEED_FACTOR = 1
SMOOTHING_FACTOR = 0.0
DEAD_ZONE = 300
MAX_STEP = 700

# Charger le modèle ONNX
onnx_model_path = "best.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Vérification du support GPU
print("Fournisseurs disponibles :", ort.get_available_providers())

# Paramètres de détection
CONFIDENCE_THRESHOLD = 0.5

# Taille d'entrée du modèle (doit correspondre au modèle)
frame_width, frame_height = 640, 640

# Capture d'écran
sct = mss.mss()
def get_screen():
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

# Fonction pour déplacer la souris avec pyautogui
def send_mouse_input(dx, dy):
    pyautogui.moveRel(dx, dy, duration=0.01)

def calculate_dx_dy(center_x, center_y, screen_width, screen_height):
    dx = center_x - (screen_width // 2)
    dy = center_y - (screen_height // 2)
    dx = max(min(dx, MAX_STEP), -MAX_STEP)
    dy = max(min(dy, MAX_STEP), -MAX_STEP)
    return dx, dy

# Variable de contrôle du déplacement de la souris
mouse_control_active = False

while True:
    frame = get_screen()

    # Appliquer letterbox pour adapter au modèle
    image, ratio, (dx, dy) = letterbox(frame, (frame_width, frame_height))
    blob = image.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    # Exécuter l'inférence avec ONNX
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    detections = outputs[0]

    for detection in detections[0]:
        confidence = detection[4]
        if confidence > CONFIDENCE_THRESHOLD:
            x_center, y_center, width, height = detection[:4]
            x1 = int((x_center - width / 2) * frame_width)
            y1 = int((y_center - height / 2) * frame_height)
            x2 = int((x_center + width / 2) * frame_width)
            y2 = int((y_center + height / 2) * frame_height)

            x1, x2 = int((x1 - dx) / ratio), int((x2 - dx) / ratio)
            y1, y2 = int((y1 - dy) / ratio), int((y2 - dy) / ratio)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if mouse_control_active:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                dx, dy = calculate_dx_dy(center_x, center_y, frame.shape[1], frame.shape[0])
                send_mouse_input(dx, dy)

    cv2.imshow("YOLOv8 ONNX Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        mouse_control_active = not mouse_control_active
        print(f"Contrôle de la souris {'activé' if mouse_control_active else 'désactivé'}")

cv2.destroyAllWindows()
