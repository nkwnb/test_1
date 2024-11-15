

## YOLO-Magic

---
### 🚀 更新与支持
- **YOLOv10**: 现已完整支持训练、推理、导出功能，且与原论文保持一致。
- **YOLOv11**: 支持训练、推理、导出。

### 📁 文件结构使用说明

1. **检测任务配置**: 配置文件位于 `YOLOv8-Magic/ultralytics-8.3.12/ultralytics/cfg/models/v8/detect`
2. **分割任务配置**: 配置文件位于 `YOLOv8-Magic/ultralytics-8.3.12/ultralytics/cfg/models/v8/segment`
3. **关键点识别配置**: 配置文件位于 `YOLOv8-Magic/ultralytics-8.3.12/ultralytics/cfg/models/v8/pose`
4. **旋转框识别配置**: 配置文件位于 `YOLOv8-Magic/ultralytics-8.3.12/ultralytics/cfg/models/v8/obb`
5. **模块代码文件**: 位于 `YOLOv8-Magic/ultralytics-8.3.12/ultralytics/nn/modules/layers`
6. **可视化工具**: 位于 `YOLOv8-Magic/ultralytics-8.3.12/ultralytics/tools`

大部分模块已集成，无需手动添加，可直接使用。

---

### ⚙️ 初始配置
首次使用需配置 GitHub 密钥，以便代码拉取。[视频教程-哔哩哔哩](https://www.bilibili.com/video/BV1bx4y1k77u/?spm_id_from=333.999.0.0&vd_source=492db9ae061509a1d7d648d975c1e77c)

#### 配置 GitHub 密钥
```bash
git config --global user.name "your name"
git config --global user.email "your_email@youremail.com"
ssh-keygen -t rsa
cd ~/.ssh
cat id_rsa.pub
```

---

### 📥 克隆项目
```bash
git clone -b v8.3.12 https://github.com/YOLOv8-Magic/YOLOv8-Magic.git
cd YOLOv8-Magic/ultralytics-v8.3.12
```

---

### 🌐 创建运行环境
```bash
conda create --name yolov8-magic python=3.10 -y
conda activate yolov8-magic
cd YOLOv8-Magic/ultralytics-8.3.12
python -m pip install --upgrade pip
pip install -e .
```

---

### 🔧 常用脚本
**推理**
```bash
python detect.py
```
**训练**
```bash
python train.py
```
**验证**
```bash
python val.py
```
**测试**
```bash
python test.py
```
**导出**
```bash
python export.py
```

---

### 🔄 拉取最新代码
保持代码最新状态：
```bash
git config pull.rebase true
git pull
```

---
