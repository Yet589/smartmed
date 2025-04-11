from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO("yolov8n.pt")  # ou ton propre modèle

# Démarrer la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détection YOLO
    results = model(frame)[0]

    # Annoter l'image avec les boîtes + labels
    annotated_frame = results.plot()

    # Obtenir les noms des objets détectés
    names = model.names
    detected_classes = results.boxes.cls.tolist() if results.boxes else []
    labels = [names[int(cls)] for cls in detected_classes]

    # Nombre total d'objets
    count = len(labels)

    # Afficher le texte sur l'image
    cv2.putText(annotated_frame, f"Objets détectés : {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if labels:
        cv2.putText(annotated_frame, f"{', '.join(labels)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Afficher la fenêtre
    cv2.imshow("Détection instruments", annotated_frame)

    # Quitter avec Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
