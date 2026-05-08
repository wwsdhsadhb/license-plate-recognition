# 🚗 License Plate Recognition — End-to-End Pipeline

基于 PyTorch 复现的中文车牌端到端识别系统，集成车牌检测（YOLOv5）与字符识别（CRNN + CTC）。

## 📌 项目背景

传统停车/门禁场景依赖人工录入，在光照不均、遮挡、多角度等复杂条件下识别率低。本项目构建了一套完整的两阶段 AI Pipeline：

1. **检测阶段**：YOLOv5 定位车牌区域 (bounding box)
2. **识别阶段**：CRNN + CTC Loss 进行字符序列识别
3. **后处理**：省份简称校验 + 格式规范化

## 🏗️ 项目结构

```
license_plate_recognition/
├── models/
│   ├── detector.py        # YOLOv5-based 车牌检测模型
│   ├── crnn.py            # CRNN 字符识别模型
│   └── pipeline.py        # 端到端 Pipeline 封装
├── utils/
│   ├── dataset.py         # 数据集加载与增强
│   ├── ctc_decoder.py     # CTC 解码器
│   ├── postprocess.py     # 后处理与格式校验
│   └── visualize.py       # 可视化工具
├── scripts/
│   ├── train_detector.py  # 检测模型训练
│   ├── train_crnn.py      # 识别模型训练
│   └── evaluate.py        # 测试集评估
├── data/
│   └── samples/           # 示例图片
├── inference.py           # 单张图片推理入口
├── requirements.txt
└── README.md
```

## ⚙️ 环境依赖

```bash
pip install -r requirements.txt
```

| 依赖 | 版本 |
|------|------|
| Python | 3.8+ |
| PyTorch | 1.12+ |
| torchvision | 0.13+ |
| OpenCV | 4.5+ |
| numpy | 1.21+ |

## 🚀 快速开始

### 推理单张图片

```bash
python inference.py --image data/samples/test.jpg --show
```

### 训练识别模型

```bash
python scripts/train_crnn.py --data_dir ./data --epochs 50 --batch_size 64
```

### 评估

```bash
python scripts/evaluate.py --weights weights/crnn_best.pth --data_dir ./data/test
```

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 字符识别准确率 | 92.3% |
| 单张推理延迟 | ~28ms (CPU) |
| 测试集车牌准确率 | 89.7% |

## 🔬 核心模型说明

### 检测模型 (YOLOv5-based)
- Backbone: CSPDarknet
- 输入尺寸: 640×640
- 输出: 车牌 bounding box + 置信度

### 识别模型 (CRNN)
- CNN 特征提取：VGG-like backbone
- RNN 序列建模：双向 LSTM × 2层
- 输出：CTC 序列解码
- 字符集：省份简称 + 字母 + 数字，共 68 类

## 📖 参考

- [LPRNet](https://arxiv.org/abs/1806.10447)
- [CRNN](https://arxiv.org/abs/1507.05717)
- [YOLOv5](https://github.com/ultralytics/yolov5)
