# v1 烟雾测试 · 药店监控视频 + qwen3-vl-plus

第一版目标：验证 qwen3-vl-plus 能不能看懂药店监控视频并正确回答问题。
链路为「前端上传视频 + 自定义问题 → 后端调用 qwen3-vl-plus → 返回结构化结果」。
本版只做视频理解，不接 ASR / YOLO / 账单对齐。

## 目录结构

```
v1_smoke_test/
├── backend/
│   ├── main.py          # FastAPI：/api/health、/api/analyze、/ 静态托管
│   ├── qwen_client.py   # qwen3-vl-plus 调用封装、提示词、JSON 解析、错误处理
│   ├── config.py        # 读取 .env / 环境变量
│   └── requirements.txt
├── frontend/
│   └── index.html       # 单页：视频选择 + 问题输入 + 结果展示（零依赖）
├── samples/
│   └── 药店视频15秒.mp4  # 一键试跑样例
├── .env.example
└── README.md
```

## 环境要求

- Python 3.10+（本项目实际在 Python 3.14 + venv 验证通过）
- 依赖：fastapi、uvicorn[standard]、openai>=1.30、python-dotenv、python-multipart

## 启动方式

1. 安装依赖（若尚未安装）：
   ```powershell
   cd "D:\John的文件\claude-test\药店监控AI\v1_smoke_test\backend"
   & "D:\John的文件\claude-test\venv\Scripts\python.exe" -m pip install -r requirements.txt
   ```

2. 配置 API Key：复制 `.env.example` 为 `.env`，填入 DashScope API Key：
   ```
   DASHSCOPE_API_KEY=sk-...
   QWEN_VL_MODEL=qwen3-vl-plus
   QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

3. 启动服务（在 backend 目录下）：
   ```powershell
   cd "D:\John的文件\claude-test\药店监控AI\v1_smoke_test\backend"
   & "D:\John的文件\claude-test\venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8001
   ```
   看到 `Uvicorn running on http://127.0.0.1:8001` 即成功。

4. 浏览器打开 http://127.0.0.1:8001/ ，选择 `samples/药店视频15秒.mp4`，
   问题可留空（走默认提示词）或自行输入，点击「开始分析」。

## 接口说明

- `GET /api/health` → `{"status":"ok","model":"qwen3-vl-plus","has_api_key":true,"max_video_mb":50,"fps":2.0}`
- `POST /api/analyze`（multipart/form-data）
  - 字段：`video`（UploadFile）、`question`（str，可空）
  - 限制：≤ 50 MB；MIME 限 mp4 / mov / avi
  - 成功返回：`{ "ok":true, "scene_summary":"...", "answer":"...", "risk_hint":"none|suspect|violation", "raw_observations":["..."], "elapsed_ms":int, "tokens":{...} }`
  - 失败返回：`{ "ok":false, "error_code":"...", "message":"...", "elapsed_ms":int }`

## 提示词设计（写死在 qwen_client.py）

- 角色：医保基金稽查 AI 助手，专门解读药店监控画面。
- 任务：先输出 3-6 句「场景说明」（人员、区域、关键动作、疑似违规线索）；
  再针对用户问题作答；最后给出 risk_hint 粗判。
- 输出强制 JSON：scene_summary / answer / risk_hint / raw_observations。
- 默认问题（用户留空时）：「请描述这段药店监控视频里发生了什么，并指出是否存在医保违规嫌疑。」
- 调用参数：fps=2、temperature=0.2、timeout=120s、response_format=json_object。

## 已验证结果（2026-06-17）

用项目自带 `药店视频15秒.mp4` 真实调用 qwen3-vl-plus，两次测试均成功：

**用例 1（问题留空，走默认提示词）** —— 耗时 46.7s，tokens 9098：
- scene_summary：识别出药店收银台、一男店员一女顾客、顾客递出疑似医保卡、
  店员多次弯腰从柜台下方取物并替换商品、全程无 POS 机/扫码动作。
- risk_hint：suspect（疑似串换）。
- raw_observations：6 条帧级线索，含时间戳（如「约第3-4秒，店员从柜台下方
  取出黄色包装药品，同时将原递来药品移入柜台下」）。

**用例 2（自定义问题）** —— 耗时 47.9s，tokens 9164：
- answer：明确回答"未观察到正规结算流程，无法确认是否真实使用医保卡结算，
  是否存在违规需结合后台交易数据进一步核查"——判断审慎，不臆断。

结论：qwen3-vl-plus 完全看懂了药店监控视频内容，能识别人物、动作、物品替换
等关键细节，并能针对自定义问题给出贴合画面的回答。烟雾测试通过。

## 已知边界

- v1 不接 ASR 语音、不接 YOLO 粗筛、不接账单对齐，无登录/历史/PDF 卷宗。
- API Key 通过本地 .env 注入，不入库（.env 已在 .gitignore）。
- 视频上限 50 MB、fps=2、temperature=0.2、timeout=120s 为默认值，可后续调参。
- 单次调用约 45-50s（含视频上传 + 模型推理），前端有 loading 计时。
