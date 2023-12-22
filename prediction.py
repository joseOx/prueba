import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    img_name="1"
    img = cv2.imread(f"datasets/images/test/{img_name}.png")

    model = YOLO("runs/detect/train5/weights/best.pt")
    #solo retorna un valor pero al ser una lista debe acceder al elemento cero
    pred = model.predict(img)[0]

    pred =pred.plot()

    cv2.imwrite(f"{img_name}.png", pred)