# text2image

基于 **阿里云百炼通义万相** 的文生图服务，配套 **DeepSeek** 做提示词改写与图片质检。
提供命令行工具与 FastAPI HTTP 接口两套入口。

## 特性

- **文生图**：调用 `wan2.6-t2i`（DashScope 同步接口），一次请求直接返回图片。
- **提示词改写**：用 DeepSeek 把中文/简短描述改写成高质量英文提示词，并给出负面提示词。
- **图片质检**：用 DeepSeek 多模态能力对生成结果打分、列出问题、给出改进建议。
- **图片落盘**：生成结果自动下载到本地（官方 URL 仅保留 24 小时），并记录 JSONL 元数据。
- **引擎抽象**：`BaseEngine` 之下可切换真实后端与离线 mock 后端，方便测试与省钱。
- **本地限流**：按 RPM 平滑放行请求，避免触发服务端限流。
- **FastAPI 服务**：异步任务提交 + 状态轮询 + 图片获取。

## 目录结构

```
text2image/
├── pyproject.toml          # 依赖与工具配置（uv 管理）
├── uv.toml                 # 把 uv 缓存固定到 D 盘
├── .env.example            # 环境变量模板
├── text2image/
│   ├── config.py           # 集中配置（pydantic-settings）
│   ├── cli.py              # 命令行入口（gen / batch / serve）
│   ├── core/
│   │   ├── schemas.py      # 数据模型
│   │   └── errors.py       # 异常体系
│   ├── engines/
│   │   ├── base.py         # 出图引擎抽象
│   │   ├── wanx.py         # 通义万相实现
│   │   └── mock.py         # 离线占位图实现
│   ├── llm/
│   │   ├── client.py       # DeepSeek（OpenAI 兼容）客户端
│   │   ├── prompt_rewriter.py
│   │   └── image_reviewer.py
│   ├── services/
│   │   ├── generator.py    # 主流程编排
│   │   └── downloader.py   # 异步下载
│   ├── storage/
│   │   ├── paths.py        # 落盘命名规则
│   │   └── repository.py   # JSONL 元数据
│   ├── api/                # FastAPI 服务
│   └── utils/              # 日志、限流
├── tests/                  # pytest 测试（全部打桩，不联网）
└── output/                 # 生成结果（gitignore）
```

## 快速开始

### 1. 环境

需要 Python 3.11+ 与 uv。

```powershell
uv sync
```

### 2. 配置

`DASHSCOPE_API_KEY` 与 `DEEPSEEK_API_KEY` 会自动从环境变量读取；
如果进程环境块被裁剪，代码还会回退到 Windows 注册表读取。

也可以复制 `.env.example` 为 `.env` 使用：

```powershell
Copy-Item .env.example .env
```

## 使用方式

### 命令行

```powershell
# 单张，带改写与质检
uv run python -m text2image.cli gen "夕阳下的海边灯塔，海浪拍打礁石" --rewrite --review

# 多张 + 指定尺寸
uv run python -m text2image.cli gen "赛博朋克城市夜景" --n 2 --size 1024*1024

# 批量（每行一条提示词，# 为注释）
uv run python -m text2image.cli batch prompts.txt --review

# 不花钱的冒烟测试
uv run python -m text2image.cli gen "test" --backend mock --no-rewrite --no-review

# 以 JSON 输出，便于脚本处理
uv run python -m text2image.cli gen "一只猫" --json
```

退出码：质检全部通过为 `0`，有图片质检未通过为 `2`，其余错误为 `1`。

### HTTP 服务

```powershell
uv run python -m text2image.cli serve --port 8000
# 或
uv run uvicorn text2image.api.app:app --host 127.0.0.1 --port 8000
```

交互式文档：http://127.0.0.1:8000/docs

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| POST | `/api/v1/images` | 提交任务，202 返回 `task_id` |
| GET | `/api/v1/tasks?limit=20` | 列出最近任务 |
| GET | `/api/v1/tasks/{task_id}` | 查询任务状态与结果 |
| GET | `/api/v1/images/{task_id}/{index}` | 获取图片文件 |

```powershell
$body = @{ prompt = "雨夜霓虹灯下的唱片店"; n = 1; rewrite = $true; review = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/images ^
  -ContentType "application/json; charset=utf-8" ^
  -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

### Python 调用

```python
import asyncio
from text2image.config import get_settings
from text2image.core.schemas import GenParams, GenRequest
from text2image.services.generator import build_service


async def main() -> None:
    service = build_service(get_settings())
    try:
        outcome = await service.generate(
            GenRequest(prompt="一只在太空的橘猫", params=GenParams(n=1)),
            rewrite=True,
            review=True,
        )
        print([str(path) for path in outcome.images])
    finally:
        await service.aclose()


asyncio.run(main())
```

## 主要配置项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 必填 | 阿里云百炼 API Key |
| `DASHSCOPE_BASE_URL` | https://dashscope.aliyuncs.com | 国内版地址 |
| `IMAGE_BACKEND` | wanx | `wanx` 真实出图 / `mock` 离线占位图 |
| `IMAGE_MODEL` | wan2.6-t2i | 出图模型 |
| `IMAGE_SIZE` | 1024*1024 | 默认尺寸（总像素 512x512 – 2048x2048） |
| `IMAGE_COUNT` | 1 | 默认张数 |
| `IMAGE_RPM` | 5 | 本地限流：每分钟请求数 |
| `IMAGE_MAX_CONCURRENCY` | 3 | 服务并发出图上限 |
| `DEEPSEEK_API_KEY` | 空 | 用于改写与质检 |
| `DEEPSEEK_MODEL` | deepseek-flash | 同时支持文本与图片输入 |
| `ENABLE_PROMPT_REWRITE` | false | 是否默认改写 |
| `ENABLE_IMAGE_REVIEW` | false | 是否默认质检 |
| `REVIEW_THRESHOLD` | 60 | 质检通过分数线 |
| `LLM_MAX_TOKENS` | 4096 | 大模型单次最大输出 |
| `T2I_OUTPUT_DIR` | output | 图片输出目录 |
| `T2I_LOG_DIR` | logs | 日志目录 |
| `LOG_LEVEL` | INFO | 日志级别 |

## 测试

```powershell
uv run pytest          # 48 个用例，全部打桩，不联网、不产生费用
uv run ruff check .    # 代码检查
uv run ruff format .   # 格式化
uv run mypy text2image # 类型检查
```

## 注意事项

- **中文路径**：项目采用扁平布局且不把自身安装进虚拟环境。若使用 src 布局 + editable 安装，
  editable 生成的 .pth 文件内容含非 ASCII 路径，而 Python 的 site 模块按系统 ANSI 编码读取，
  会在解释器启动时直接崩溃（UnicodeDecodeError: 'gbk' codec），且 PYTHONUTF8=1 无法修复。
- **图片 URL 有效期 24 小时**，必须及时下载；本项目默认自动落盘。
- **费用**：wan2.6-t2i 约 ¥0.2/张（1024x1024）；n 直接决定费用，测试时建议 n=1。
- **任务状态存在内存中**，服务重启即丢失；图片与 metadata.jsonl 仍在磁盘。
- 若未配置 DeepSeek Key，改写与质检会自动跳过，不影响出图。

## 已知限制

- 任务队列为进程内实现，多实例部署需要外部队列（Redis / Celery）。
- 未实现图片重试补偿与断点续跑。
- mock 引擎生成的占位图仅用于链路验证。

## License

MIT
