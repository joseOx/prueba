from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    model.train(data="train.yaml", epochs = 30 , imgsz=(640,640), batch = 4, optimizer="Adam")