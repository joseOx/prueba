from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    model.train(data="train.yaml", epochs = 100 , imgsz=(640,640), batch = 32, optimizer="Adam")