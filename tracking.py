import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort

if __name__ == '__main__':
    cap = cv2.VideoCapture("videos/tablero_video.mp4")

    model = YOLO("runs/detect/train5/weights/best.pt")

    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        results = model(frame, stream=True)

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            class_values = res.boxes.data.cpu().numpy()[:, -1]
            #print("Atributos y métodos de res.boxes:", classes)
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            class_names = {
                0: "chess-pieces",
                1: "bishop",
                2: "black-bishop",
                3: "black-king",
                4: "black-knight",
                5: "black-pawn",
                6: "black-queen",
                7: "black-rook",
                8: "white-bishop",
                9: "white-king",
                10: "white-knight",
                11: "white-pawn",
                12: "white-queen",
                13: "white-rook"
            }
            
            for idx, track in enumerate(tracks):
                xmin, ymin, xmax, ymax, track_id = track
                class_value = class_values[idx]
                class_name = class_names.get(class_value, "Desconocido")
                cv2.putText(img=frame, text=f"Clase: {class_name}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        #frame = results[0].plot()

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()