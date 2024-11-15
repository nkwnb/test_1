from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    cfg_path = 'D:/Project/YOLOnew/YOLOv8-Magic/ultralytics-8.3.12/ultralytics/cfg/models/myModels/yolov8-PSA-AFPN.yaml'

    model = YOLO(cfg_path)
    
    model._new(cfg_path, task='detect', verbose=True)