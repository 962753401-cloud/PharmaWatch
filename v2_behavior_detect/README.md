# 药店监控AI v2 — 行为特征主动发现系统

## 概述

使用 YOLO-World 开放词汇检测 + Supervision ByteTrack 人员跟踪，倍速扫描长视频。
监测到特定的"高危行为特征"后，立即用 ffmpeg 切片保存。

### 三个触发器

| 触发器 | 条件 | 药店场景映射 |
|--------|------|-------------|
| 人群聚集 (crowd) | 连续 ≥5 帧每帧 ≥6 人 | 该区域同时出现人数 ≥3 人 |
| 人员停留 (loiter) | 同一 tracker_id 在 50px 内停留 ≥20s | 单人在收银台停留超过3分钟 |
| 携带物品 (carry) | 人物框扩展区域内检测到物体 | 持卡数 ≥2 |

触发后：以当前时间为中心前后各切60秒，总时长≤120秒，保存 MP4 到后台。
同一触发器60秒冷却期。

### YOLO-World 可用性评估

标准 COCO YOLO（含 YOLO11）仅覆盖"人"（5个药店目标中1个，20%）。
手、医保卡、药盒、POS机全部不在 COCO 80类中。

YOLO-World 是唯一支持自定义类别的方案。实测在真实药店视频上：
- 人 (person): 置信度 0.5-0.91，可靠
- 药盒 (box): 置信度 0.4-0.44，中等
- 药瓶 (bottle): 置信度 0.6-0.62，中等
- 手 (hand): 置信度 0.1-0.14，不可靠
- 医保卡 (bank card): 置信度 0.10，不可靠
- POS机: 置信度 0.0，失败

结论：YOLO-World 适合"行为特征层面"检测（人数/聚集/停留/携物），
不适合精细物体识别（手/卡/POS机需后续版本通过微调解决）。

## 启动方法

### 1. 启动后端服务器

双击 `start_server.bat`，或在终端执行：

```powershell
cd D:\John的文件\claude-test\药店监控AI\v2_behavior_detect\backend
D:\John的文件\claude-test\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8001
```

### 2. 打开前端页面

浏览器访问 `http://127.0.0.1:8001`

### 3. 输入视频

在"本地视频路径"输入框中填入视频完整路径，例如：

```
D:\John的文件\claude-test\药店监控AI\户外视频-30min.mp4
```

也可以用拖拽方式上传视频文件。

### 4. 查看结果

点击"开始分析"后，页面显示扫描进度条和已发现触发数。
扫描完成后，结果区列出所有切片，可内嵌播放和下载。

## 测试结果（10分钟户外视频）

- 扫描耗时：约225秒（10分钟视频，2fps采样）
- 触发器命中：17个（10个携带物品 + 7个人群聚集）
- 切片文件：17个 MP4，每个≤120秒
- 人工验证：切片画面包含人群聚集、携带背包/手提包等行为特征

## 文件结构

```
v2_behavior_detect/
├── start_server.bat          # 一键启动脚本
├── README.md
├── backend/
│   ├── main.py               # FastAPI: 端点、任务管理、静态服务
│   ├── config.py             # 配置：权重路径、触发参数、阈值
│   ├── scanner.py            # YOLO-World + Supervision 扫描 + 触发逻辑
│   ├── slicer.py             # ffmpeg 切片
│   ├── requirements.txt
│   ├── clips/                # 切片输出目录
│   └── uploads/              # 上传视频目录
└── frontend/
    └── index.html            # 单文件前端
```

## 技术参数

- 模型：yolov8s-world.pt（开放词汇检测）
- 检测类别：person, hand, card, box, bottle, bag, backpack, handbag, phone
- 采样：每15帧取1帧（30fps视频→2fps，约15倍速）
- 跟踪：Supervision ByteTrack（person 类）
- 切片：ffmpeg stream copy（-c copy，无损快速）
- 运行环境：Python 3.14 + CPU推理（无GPU）