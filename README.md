# 药店监控 AI · 医保基金监管系统

> 基于多模态大模型的药店监控视频/音频分析系统，用于事后审计、检测医保违规行为。
> 三个迭代版本，从小步验证到完整闭环，每一步都可独立运行、独立验证。

[English](./README_EN.md) | 中文

---

## 项目背景

本项目是一个面向医保基金监管场景的 AI 云眼系统原型。系统以药店监控视频和语音为输入，通过多模态大模型分析，检测药物串换、违规套现、多卡同刷、超量配药等违规行为。

采用"数据驱动倒查、本地高速粗筛、云端大模型精审"的离线批处理架构，将大模型 API 调用成本压缩 90% 以上。

## 三版演进路线

| 版本 | 名称 | 核心能力 | 验证目标 |
|------|------|----------|----------|
| v1 | 烟雾测试 | qwen3-vl-plus 视频问答 | 大模型能否看懂药店监控画面 |
| v2 | 行为特征发现 | YOLO-World + Supervision 视觉触发 | 本地倍速扫描能否精准定位高危片段 |
| v3 | 音频敏感词盲扫 | 语音先行 + ASR + 两阶段风控漏斗 | 音频管线能否定位敏感词并精准切取视频 |

每个版本都是独立可运行的 FastAPI + 单页前端应用，复用上传组件，后端逻辑逐版演进。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│  数据源：药店监控视频 (MP4) + 语音 (WAV) + 医保结算流水      │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     v1 视频问答   v2 视觉触发   v3 音频先行
     (qwen3-vl)   (YOLO-World)  (ASR + 风控)
          │            │            │
          ▼            ▼            ▼
     结构化判定   行为触发切片   敏感词定位切片
          │            │            │
          └────────────┴────────────┘
                       │
                       ▼
              前端展示 + 证据片段
```

## 快速开始

### 环境要求

- Python 3.10+（项目在 Python 3.14 验证通过）
- FFmpeg / FFprobe（加入系统 PATH）
- [通义千问 DashScope API Key](https://dashscope.aliyuncs.com/)
- v2 额外需要：YOLO-World 权重文件 `yolov8s-world.pt`

### 通用安装

```powershell
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活
.\venv\Scripts\Activate.ps1

# 3. 进入对应版本目录安装依赖
cd v1_smoke_test\backend
pip install -r requirements.txt
```

### 运行 v1（烟雾测试）

```powershell
cd v1_smoke_test\backend
# 配置 API Key
copy ..\.env.example ..\.env
# 编辑 ..\.env 填入 DASHSCOPE_API_KEY
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```
浏览器打开 http://127.0.0.1:8001

### 运行 v2（行为特征发现）

```powershell
cd v2_behavior_detect\backend
pip install -r requirements.txt
# 将 yolov8s-world.pt 放入 ..\..\weights\ 目录
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```
浏览器打开 http://127.0.0.1:8001

### 运行 v3（音频敏感词盲扫）

```powershell
cd v3_audio_scan\backend
pip install -r requirements.txt
copy ..\.env.example ..\.env
# 编辑 ..\.env 填入 DASHSCOPE_API_KEY
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```
浏览器打开 http://127.0.0.1:8002

### 一键启动（Windows）

双击 `start_all.bat`，选择要启动的版本。

## 目录结构

```
版本一/
├── README.md                      # 本文件（中文）
├── README_EN.md                   # 英文说明
├── LICENSE                        # MIT 开源协议
├── .gitignore
├── start_all.bat                  # 一键启动菜单
├── docs/
│   ├── 原始业务需求.txt             # 业务需求文档
│   ├── 技术架构设计.txt             # 技术架构设计方案
│   └── 开发过程.txt                 # 开发迭代记录
├── v1_smoke_test/                 # 版本一：烟雾测试
│   ├── .env.example
│   ├── .gitignore
│   ├── backend/
│   │   ├── main.py                # FastAPI 入口
│   │   ├── qwen_client.py         # qwen3-vl-plus 调用封装
│   │   ├── config.py              # 配置加载
│   │   └── requirements.txt
│   └── frontend/
│       └── index.html             # 单页前端
├── v2_behavior_detect/            # 版本二：行为特征发现
│   ├── .gitignore
│   ├── start_server.bat
│   ├── backend/
│   │   ├── main.py                # FastAPI + 任务管理
│   │   ├── scanner.py             # YOLO-World 扫描 + 触发逻辑
│   │   ├── slicer.py              # FFmpeg 切片
│   │   ├── config.py              # 触发阈值配置
│   │   └── requirements.txt
│   └── frontend/
│       └── index.html
└── v3_audio_scan/                 # 版本三：音频敏感词盲扫
    ├── .env.example
    ├── .gitignore
    ├── start_server.bat
    ├── backend/
    │   ├── main.py                # FastAPI 入口
    │   ├── scanner.py             # 管线编排
    │   ├── audio_pipeline.py      # 音轨提取 + Silero VAD
    │   ├── asr_client.py          # Fun-ASR WebSocket 客户端
    │   ├── keyword_matcher.py     # pypinyin 关键词匹配
    │   ├── risk_llm.py            # qwen-plus 风控判定
    │   ├── slicer.py              # 边界限幅切片
    │   ├── config.py              # 配置加载
    │   └── requirements.txt
    └── frontend/
        └── index.html
```

## 各版本技术详情

### v1 烟雾测试 · qwen3-vl-plus 视频问答

**目标**：验证 qwen3-vl-plus 能否看懂药店监控视频并正确回答问题。

**链路**：前端上传视频 + 自定义问题 → 后端调用 qwen3-vl-plus → 返回结构化 JSON。

**输出字段**：
- `scene_summary`：3-6 句场景描述
- `answer`：针对用户问题的回答
- `risk_hint`：风险粗判（none / suspect / violation）
- `raw_observations`：帧级线索列表

**实测结果**：模型成功识别收银台、店员顾客、递卡、弯腰取物、商品替换等细节，risk_hint 判定为 suspect（疑似串换）。

### v2 行为特征发现 · YOLO-World + Supervision

**目标**：本地倍速扫描长视频，监测到高危行为特征后自动切片。

**三个触发器**：

| 触发器 | 条件 | 药店场景映射 |
|--------|------|-------------|
| 人群聚集 crowd | 连续帧每帧人数 ≥ 阈值 | 收银区多人聚集 |
| 携带物品 carry | 人物框区域检测到目标物体 | 持卡 / 拿手机 |
| 停留坐姿 loiter | 同一 tracker 在座位区域停留 ≥ N 秒 | 收银台长时间逗留 |

**技术要点**：
- YOLO-World 开放词汇检测（支持自定义类别）
- Supervision ByteTrack 人员跟踪
- 每 15 帧采样 1 帧（约 15 倍速）
- 触发后 FFmpeg 重新编码切片（帧精确）
- 前端实时预览 YOLO 标注帧

**YOLO-World 可用性评估**：适合行为特征层面检测（人数/聚集/停留/携物），不适合精细物体识别（手/卡/POS 机置信度过低）。

### v3 音频敏感词盲扫 · 语音先行

**目标**：让声音数据先行，通过 ASR + 两阶段风控漏斗定位敏感词，精准切取视频片段。

**管线流程**：

```
长视频(MP4)
  │  FFmpeg 提取音轨 (16kHz mono PCM)
  ▼
大 WAV
  │  按 5 分钟窗口 → Silero VAD 预筛
  │  人声 < 5s 的窗口跳过（节约 ASR token）
  ▼
有人声的 5 分钟分片
  │  Fun-ASR Realtime (WebSocket) → 词级时间戳
  │  换算: T_absolute = chunk_start + word_begin_ms / 1000
  ▼
Stage 1: pypinyin 拼音精确匹配
  │  关键词词库命中 → 标记疑似违规 + 绝对时间戳
  ▼
Stage 2: qwen-plus 上下文风控
  │  命中词前后上下文 → LLM 判定 确认/暗语/误报
  ▼
FFmpeg 边界限幅切片
  Start = max(0, T_absolute - slice_half)
  End   = min(duration, T_absolute + slice_half)
```

**关键技术**：
- Silero VAD 人声预筛（跳过静音片段，节省 90% ASR 成本）
- Fun-ASR 词级时间戳（毫秒精度）
- pypinyin 拼音匹配（容忍 ASR 误识）
- 两阶段风控漏斗（关键词匹配 + LLM 上下文复核）
- 边界限幅切片（防止负数或越界）
- 冷却期去重（10 秒内合并重复命中）

**演示模式**（DEMO_MODE=1）：生产词库照常运行，同时增补"任意人声触发"规则，用于在非药店视频上验证时间戳换算与视频对齐。

## API 接口

三个版本共享相似的 API 设计：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查，返回配置信息 |
| POST | `/api/analyze` | 上传视频文件分析（multipart） |
| POST | `/api/analyze/path` | 通过本地路径分析（v2/v3） |
| GET | `/api/task/{task_id}` | 查询任务状态（v2/v3） |
| GET | `/api/task/{task_id}/transcript` | 查询转写快照（v3） |
| GET | `/api/clips/{filename}` | 获取切片视频 |
| GET | `/api/task/{task_id}/frame/{idx}` | 获取标注帧（v2） |
| GET | `/` | 前端页面 |

## 配置说明

各版本通过 `.env` 文件配置（参考 `.env.example`）：

| 变量 | 版本 | 说明 |
|------|------|------|
| `DASHSCOPE_API_KEY` | v1/v3 | 通义千问 API Key |
| `QWEN_VL_MODEL` | v1 | 视觉模型名（默认 qwen3-vl-plus） |
| `MAX_VIDEO_MB` | v1 | 视频大小上限（默认 50） |
| `WEIGHTS_PATH` | v2 | YOLO 权重路径 |
| `CROWD_THRESHOLD` | v2 | 人群触发人数阈值 |
| `ASR_MODEL` | v3 | ASR 模型名 |
| `LLM_MODEL` | v3 | 风控 LLM 模型（默认 qwen3.5-flash） |
| `DEMO_MODE` | v3 | 演示模式开关 |
| `MOCK_ASR` | v3 | Mock ASR 开关（无配额时用） |
| `SLICE_HALF` | v3 | 切片半宽秒数（默认 10） |

## 技术栈

- **后端**：Python 3.10+、FastAPI、Uvicorn
- **大模型**：通义千问 qwen3-vl-plus、qwen-plus、Fun-ASR
- **视觉检测**：YOLO-World (Ultralytics)、Supervision (ByteTrack)
- **音频处理**：FFmpeg、Silero VAD、pypinyin
- **前端**：原生 HTML/CSS/JS（零依赖单页应用）

## 已知限制

- v1 不接 ASR 语音、不接 YOLO 粗筛、不接账单对齐
- v2 YOLO-World 对医保卡、药盒等精细物体置信度偏低
- v3 依赖网络调用 ASR/LLM API，单次处理耗时取决于视频长度
- 所有版本为 CPU 推理（无 GPU），v2 扫描速度受限于 CPU 性能
- 演示数据使用户外/买菜视频替代药店监控视频

## 开发过程

完整的开发迭代记录见 [docs/开发过程.txt](./docs/开发过程.txt)，包含每个版本的需求演进、阈值调整、问题修复过程。

## 开源协议

[MIT License](./LICENSE)
