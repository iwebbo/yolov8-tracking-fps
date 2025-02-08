import onnxruntime as ort
import numpy as np
import cv2

def preprocess_image(image_path, input_size):
    image = cv2.imread(image_path)
    if image is None:  # Gestion des erreurs de lecture d'image
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    image_resized = cv2.resize(image, input_size) # Redimensionnement avant conversion de couleur
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_input = np.expand_dims(image_transposed, axis=0)
    return image, image_input  # Retourne l'image originale et l'image prétraitée


def postprocess_detections(detections, image_shape):
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections.T:
        box = detection[:4] * np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        boxes.append(box)  # Ajouter le tableau box à la liste
        confidences.append(detection[4])
        class_ids.append(np.argmax(detection[5:]))

    return np.array(boxes), np.array(confidences), np.array(class_ids)

def draw_boxes(image, boxes, scores, class_ids, class_names):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)  # Conversion en int ici, plus sûr
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    model_path = "best_fort.onnx"
    image_path = "capture.jpg"
    input_size = (640, 640)
    class_names = ["0", "1"]

    try:
        image, input_image = preprocess_image(image_path, input_size) # Récupération de l'image originale
    except ValueError as e:
        print(f"Erreur de prétraitement : {e}")
        return

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    detections = np.array([[0.1, 0.2, 0.3, 0.4, 0.9, 0, 1], [0.5, 0.6, 0.7, 0.8, 0.8, 1, 0]]).T

    boxes, confidences, class_ids = postprocess_detections(detections, image.shape)

    draw_boxes(image, boxes, confidences, class_ids, class_names) # Passage de l'image à draw_boxes

if __name__ == "__main__":
    main()