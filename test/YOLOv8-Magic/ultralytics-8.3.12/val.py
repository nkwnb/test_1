from ultralytics import YOLO

if __name__ == "__main__":
    # 加载模型
    model = YOLO("D:/Project/YOLOnew/YOLOv8-Magic/ultralytics-8.3.12/runs/train/v8_AFPN_healthy/weights/best.pt")
    # 验证模型
    metrics = model.val(
        data="D:/Project/YOLOnew/chicken.yaml",
        split="test",  # (str) 用于验证的数据集拆分，例如'val'、'test'或'train'
        batch=1,  # (int) 每批的图像数量（-1 为自动批处理）
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        device="",  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=8,  # 数据加载的工作线程数（每个DDP进程）
        save_json=False,  # 保存结果到JSON文件
        save_hybrid=False,  # 保存标签的混合版本（标签 + 额外的预测）
        conf=0.001,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
        iou=0.6,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        project="runs/val",  # 项目名称（可选）
        name="exp",  # 实验名称，结果保存在'project/name'目录下（可选）
        max_det=300,  # 每张图像的最大检测数
        half=False,  # 使用半精度 (FP16)
        dnn=False,  # 使用OpenCV DNN进行ONNX推断
        plots=True,  # 在训练/验证期间保存图像
    )

    print(f"mAP50-95: {metrics.box.map}")  # map50-95
    print(f"mAP50: {metrics.box.map50}")  # map50
    # print(f"mAP55: {metrics.box.map55}")  # map55
    # print(f"mAP60: {metrics.box.map60}")  # map60
    # print(f"mAP65: {metrics.box.map65}")  # map65
    # print(f"mAP70: {metrics.box.map70}")  # map70
    # print(f"mAP75: {metrics.box.map75}")  # map75
    # print(f"mAP80: {metrics.box.map80}")  # map80
    # print(f"mAP85: {metrics.box.map85}")  # map85
    # print(f"mAP90: {metrics.box.map90}")  # map90
    # print(f"mAP95: {metrics.box.map95}")  # map95
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}")  # FPS
