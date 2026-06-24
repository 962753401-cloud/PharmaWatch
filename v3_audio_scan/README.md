# 药店监控AI v3 - 音频敏感词盲扫触发（语音先行）

音频先行的医保违规检测管线：从长视频中提取音轨，用 Silero VAD 预筛静音片段，
将有人声的 5 分钟分片交给 Fun-ASR 词级时间戳转写，经两阶段风控漏斗
（pypinyin 关键词匹配 + qwen-plus 上下文判定）定位敏感词绝对时间戳，
按边界限幅切取前后各 1 分钟视频片段落盘。

## 管线流程

```
长视频(MP4)
  │  FFmpeg 提取音轨 (-ac 1 -ar 16000 -c:a pcm_s16le)
  ▼
大 WAV (16kHz mono)
  │  按 300s 窗口滑动 → Silero VAD 扫描
  │  人声 < 5s 的窗口直接跳过（节约 ASR token）
  ▼
有人声的 5 分钟分片 (vid_[ID]_[start]_[end].wav)
  │  Fun-ASR Realtime (WebSocket) → 词级 begin_time/end_time (ms)
  │  时间戳换算: T_absolute = chunk_start_sec + word.begin_time_ms / 1000
  ▼
Stage 1: pypinyin 关键词匹配
  │  药店词库 6 类违规 × 拼音子串/编辑距离≤1
  │  命中即标记疑似违规 + 记录绝对时间戳
  ▼
Stage 2: qwen-plus 上下文风控（不阻塞切片）
  │  命中词前后各 5 句 → LLM 判定 涉嫌/合规
  ▼
FFmpeg 边界限幅切片
  Start = max(0, T_absolute - 60)
  End   = min(total_duration, T_absolute + 60)
  → clip_[ID]_[abs_ts].mp4
```

## 触发阈值设计（药店医保场景）

词库按 6 类违规分组，每类含核心词 + 变体：

| 类型 | 关键词示例 |
|------|-----------|
| 药物串换 | 换药、串换、换成、红参、礼盒、保健品 |
| 违规套现 | 套现、套个现、弄点现金、返现、折现 |
| 多卡同刷 | 几张卡、刷两张、家人卡、一起刷 |
| 超量配药 | 多开点、囤点、多拿几盒、先存着 |
| 虚构服务 | 不用拿药、直接刷、空刷、不拿货 |
| 防监管暗语 | 那个东西、你懂的、别登记、不开票、抹掉 |

Stage 1 用 pypinyin 转拼音后做子串匹配 + 编辑距离≤1 容错，
容忍 ASR 误识（如"川换"命中"串换"）。

演示模式（DEMO_MODE=1，默认开启）：生产词库照常运行；同时增补
"任意人声触发"规则，凡 VAD 判定有人声的分片即触发切片，
用于在非药店视频上验证时间戳换算与视频对齐。

## 快速开始

### 依赖安装

```powershell
cd D:\John的文件\claude-test\药店监控AI\v3_audio_scan\backend
D:\John的文件\claude-test\venv\Scripts\python.exe -m pip install silero-vad pypinyin websocket-client python-dotenv
```

（torch / openai / fastapi 已在共享 venv 中。）

### 配置

编辑 `.env`，填写 DashScope API Key：

```
DASHSCOPE_API_KEY=sk-your-key
ASR_MODEL=fun-asr-realtime-2026-02-28
LLM_MODEL=qwen-plus
DEMO_MODE=1
MOCK_ASR=0
```

> `qwen3.5-flash` 在部分账号返回 403，系统会自动回落到 `qwen-plus`。
> `MOCK_ASR=1` 时注入合成转写（含药店违规关键词），用于无 ASR 配额时验证管线。

### 启动

```powershell
.\start_server.bat
```

或手动：

```powershell
cd backend
D:\John的文件\claude-test\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8002
```

浏览器打开 http://127.0.0.1:8002

### 使用

1. 拖拽视频文件或输入本地绝对路径
2. 点击"开始盲扫"
3. 观察管线阶段进度（提取音轨 → VAD预筛 → ASR转写 → 风控判定）
4. 右侧实时转写视图显示当前分片的流式文本
5. 命中切片卡片展示视频 + 转写片段 + 命中词 + 风控结论

## 文件结构

```
v3_audio_scan/
├── .env                    # 配置（API Key、模型、阈值）
├── .env.example
├── start_server.bat
├── README.md
├── frontend/
│   └── index.html          # 前端（上传 + 管线视图 + 切片展示）
└── backend/
    ├── config.py           # 配置加载
    ├── audio_pipeline.py   # 音轨提取 + Silero VAD 分窗
    ├── asr_client.py       # Fun-ASR WebSocket 客户端 + Mock
    ├── keyword_matcher.py  # pypinyin 关键词匹配（Stage 1）
    ├── risk_llm.py         # qwen-plus 风控判定（Stage 2）
    ├── slicer.py           # FFmpeg 边界限幅切片
    ├── scanner.py          # 管线编排 + 任务状态
    └── main.py             # FastAPI 入口
```

## 技术说明

### Silero VAD 与中文路径

venv 安装在含中文字符的路径下，`torch.jit.load` 的 C 层 fopen 无法打开
bundled 模型文件。解决方法：将 `silero_vad.jit` 复制到 ASCII 临时路径后加载。
无需 torchaudio，WAV 读取用 stdlib `wave` + numpy。

### ASR 时间戳换算

Fun-ASR 返回每个字的 `begin_time`/`end_time`（毫秒，相对分片起点）。
换算器写死在 scanner 中：`T_absolute = chunk_start_sec + begin_time_ms / 1000`。

### 边界限幅

切片前做边界限幅，防止负数或越界：
- `Start = max(0, T_absolute - 60)`
- `End = min(total_duration, T_absolute + 60)`
- 总时长上限 120s（SLICE_MAX）

已在 964s 长视频上验证：30s 命中 → [0, 90]；950s 命中 → [890, 964.5]。
