from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="D:/leyca_vision/pliers_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="alicates_v1",
    )

if __name__ == "__main__":
    main()