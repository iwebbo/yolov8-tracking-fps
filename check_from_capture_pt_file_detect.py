import torch
from ultralytics import YOLO
import cv2

def main():
    # Chemin vers votre modèle YOLOv8 entraîné
    model_path = "best_fort.pt"
    # Chemin vers l'image que vous souhaitez tester
    image_path = "capture.jpg"
    # Liste des noms de classes correspondant à votre modèle
    class_names = ["0", "1"]  # Remplacez par vos classes

    # Chargement du modèle YOLOv8
    model = YOLO(model_path)

    # Chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image à partir de {image_path}")
        return

    # Exécution de la prédiction
    results = model.predict(image)

    # Parcours des résultats et affichage des détections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Coordonnées de la boîte englobante
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Confiance de la détection
            confidence = box.conf[0]
            # ID de la classe détectée
            class_id = int(box.cls[0])
            # Nom de la classe détectée
            class_name = class_names[class_id] if class_id < len(class_names) else f"Classe {class_id}"

            # Dessiner la boîte englobante sur l'image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Ajouter le label de la classe et la confiance
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Affichage de l'image avec les détections
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()