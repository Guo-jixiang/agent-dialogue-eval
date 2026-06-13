# PromptScope - 对话指令评测平台

[English](README_EN.md)

PromptScope 是一套面向对话式 AI 产品的自动化指令遵循评测工具。通过模拟真实用户与对话模型进行多轮交互，由专业评判模型从 7 个维度打分，最终生成可量化、可追溯、可对比的评测报告。

> 一句话：把原来需要人工听几百通对话才能给出的质检结论，压缩到几分钟内自动完成。

## 核心功能

- **全自动对话模拟** - 4 维度 Persona 系统（配合意愿、话量风格、业务熟悉度、时间紧迫度），理论上可组合出 72 种不同用户行为
- **7 维度独立评分** - 流程遵循、约束合规、知识准确、风格语气、连贯性、安全合规、应变能力
- **自洽性检验** - 对主观维度进行多次独立评判，取一致性结论，降低随机性
- **异常检测** - 自动标记可疑高分、双峰分布、低置信度等异常情况
- **证据可追溯** - 每条评分结论附带对话轮次引用和原文摘录
- **3 层分析链** - 单对话分析（ReportAgent）→ 横向对比（JudgeAgent）→ 根因归因（AttributionAgent）
- **断点续跑** - 评测中途中断后自动从上次断点继续
- **多种使用方式** - 命令行、Web 控制台（Vue.js）、REST API（SSE 流式推送）
- **指令优化建议** - 基于评测结果自动生成任务指令的改进方案
- **内置 AI 助手** - 可结合报告数据回答问题，给出具体可操作的分析
- **多格式报告** - HTML（含可视化图表）、Markdown、JSON 三种格式

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│  任务指令（System Prompt）                                 │
└────────────────────┬─────────────────────────────────────┘
                     │ 阶段 1：解析
                     ▼
┌──────────────────────────────────────────────────────────┐
│  InstructionSpec（流程图、约束、知识点）                     │
└────────────────────┬─────────────────────────────────────┘
                     │ 阶段 2：生成 Persona
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Persona 矩阵（4 维度 × N 种组合）                         │
└────────────────────┬─────────────────────────────────────┘
                     │ 阶段 3：模拟 + 评测
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ 对话 1   │ │ 对话 2   │ │ 对话 N   │    （并行）
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │ 7 个评判 │ │ 7 个评判 │ │ 7 个评判 │    （每条对话并行）
    └────┬────┘ └────┬────┘ └────┬────┘
         └───────────┼───────────┘
                     │ 阶段 4：分析
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    Layer 4     Layer 5     Layer 6
    单对话分析   横向对比     根因归因
   (ReportAgent)(JudgeAgent)(AttributionAgent)
                     │
                     ▼
              ┌─────────────┐
              │   最终报告    │
              │ HTML/MD/JSON │
              └─────────────┘
```

## 评分维度

| 维度 | Key | 默认权重 | 评测内容 |
|------|-----|---------|---------|
| 流程遵循 | `flow` | 25% | 是否按任务指令规定的流程节点推进 |
| 约束合规 | `constraint` | 20% | 是否严格遵守硬性/软性约束规则 |
| 知识准确 | `knowledge` | 20% | 业务信息是否与指令中的知识要点一致 |
| 风格语气 | `style` | 10% | 表达风格是否符合角色定义 |
| 连贯性 | `coherence` | 10% | 上下文逻辑一致，无答非所问或前后矛盾 |
| 安全合规 | `safety` | 10% | 无有害内容、无系统提示泄露、无幻觉编造 |
| 应变能力 | `adaptability` | 5% | 面对异常用户行为时的处理质量 |

**硬约束惩罚机制：** 违反硬约束时采用阶梯式惩罚（1 项：x0.9，2 项：x0.8，3 项：x0.7，4+ 项：x0.6）。

## 快速上手

### 环境要求

- Python 3.11+
- Node.js 18+（构建 Web 前端）
- OpenAI 兼容的 LLM API Key

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境

复制 `.env.example` 为 `.env`，填入 API Key：

```bash
cp .env.example .env
```

```env
# 评判模型（推荐 GPT-4o 或同等能力模型）
EVAL_LLM_API_KEY=your_key
EVAL_LLM_BASE_URL=https://api.openai.com/v1
EVAL_LLM_MODEL=gpt-4o

# 用户模拟模型（可用较低成本模型）
SIM_LLM_API_KEY=your_key
SIM_LLM_BASE_URL=https://api.openai.com/v1
SIM_LLM_MODEL=gpt-4o-mini

# 被测的对话模型
AGENT_LLM_API_KEY=your_key
AGENT_LLM_BASE_URL=https://api.openai.com/v1
AGENT_LLM_MODEL=your_model_name

# AI 助手 LLM（内置助手，权限与 eval/sim key 隔离）
# 留空则自动回退到 EVAL_LLM_*；推荐配独立低成本 key（如 gpt-4o-mini）
ASSISTANT_LLM_API_KEY=
ASSISTANT_LLM_BASE_URL=https://api.openai.com/v1
ASSISTANT_LLM_MODEL=gpt-4o-mini
```

### 方式一：Web 控制台（推荐）

```bash
# 构建前端（首次运行）
cd web && npm install && npm run build && cd ..

# 启动服务
python cli.py serve --port 8000
```

浏览器打开 `http://localhost:8000`。

开发模式（前端热更新）：
```bash
python cli.py serve --port 8000 --dev
```

### 方式二：命令行

```bash
# 完整评测（使用所有 Persona）
python cli.py run --instruction my_task.txt

# 仅测试部分 Persona
python cli.py run --instruction my_task.txt --personas cooperative,resistant,impatient

# 仅解析指令结构（不运行评测）
python cli.py parse --instruction my_task.txt

# 模拟单次对话（不评分，用于调试）
python cli.py simulate --instruction my_task.txt --persona resistant --max-turns 20
```

### 方式三：REST API

```bash
# 提交评测任务（SSE 流式推送进度）
curl -N -X POST http://localhost:8000/api/eval/run \
  -H "Content-Type: application/json" \
  -d '{"instruction": "你是一名客服专员...", "agent_type": "llm"}'

# 直接评测已有对话（无需模拟）
curl -X POST http://localhost:8000/api/eval/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "你是...",
    "dialogue": [
      {"role": "agent", "content": "您好，我是小美..."},
      {"role": "user", "content": "嗯，什么事？"}
    ]
  }'

# 获取报告
curl http://localhost:8000/api/eval/report/{report_id}?fmt=html

# 列出最近报告
curl http://localhost:8000/api/eval/reports
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/eval/parse` | 解析任务指令，返回结构化 Spec |
| `POST` | `/api/eval/run` | 提交完整评测任务（SSE 流式） |
| `POST` | `/api/eval/evaluate` | 对已有对话进行评测（同步返回） |
| `POST` | `/api/eval/simulate` | 模拟单次对话（不评分） |
| `POST` | `/api/eval/generate-personas` | 基于指令 AI 生成测试 Persona |
| `POST` | `/api/eval/optimize-prompt` | 生成指令优化建议（SSE 流式） |
| `GET` | `/api/eval/reports` | 列出最近 20 份报告 |
| `GET` | `/api/eval/report/{id}` | 获取报告（`?fmt=json/html/md`） |
| `DELETE` | `/api/eval/report/{id}` | 删除报告 |
| `POST` | `/api/assistant/chat` | 内置 AI 助手（SSE 流式） |

## 项目结构

```
.
├── api/                        # FastAPI Web 服务
│   ├── app.py                  # 应用工厂
│   └── routes.py               # 全部 API 端点
├── cli.py                      # CLI 入口（Click）
├── config/
│   ├── prompts/                # Jinja2 提示词模板（17 个）
│   └── settings.py             # Pydantic 配置（读取 .env）
├── core/
│   ├── agent.py                # Agent 抽象层（LLM + Friday）
│   ├── llm_client.py           # OpenAI 兼容 LLM 客户端
│   ├── models.py               # 全部 Pydantic 数据模型
│   └── rate_limit.py           # 令牌桶限流器
├── evaluator/
│   ├── judges/                 # 7 个维度评判器 + 自洽性包装器
│   ├── anomaly.py              # 异常检测
│   ├── attribution_agent.py    # Layer 6：根因归因
│   ├── decomposer.py           # 指令 → 检查点拆解
│   ├── eval_pipeline.py        # 单对话评测流水线
│   ├── evidence.py             # 证据摘要
│   ├── judge_agent.py          # Layer 5：横向对比
│   ├── report_agent.py         # Layer 4：单对话分析
│   └── scorer.py               # 评分聚合
├── parser/
│   └── instruction_parser.py   # 基于 LLM 的指令解析
├── report/
│   ├── generator.py            # 多格式报告生成
│   └── templates/              # HTML/Markdown 报告模板
├── simulator/
│   ├── engine.py               # 对话引擎（兼容层）
│   ├── persona.py              # Persona 生成与丰富
│   ├── persona_generator.py    # AI 驱动的 Persona 生成
│   ├── runner.py               # 对话运行器
│   ├── scenario.py             # 场景矩阵执行
│   └── user_simulator.py       # 用户行为模拟
├── tests/                      # pytest 测试套件
├── web/                        # Vue.js 前端
├── Dockerfile                  # 容器构建
├── docker-compose.yml          # Docker Compose 配置
├── deploy.sh                   # 一键部署脚本
└── run_eval.py                 # 主编排流水线
```

## 配置说明

所有配置通过环境变量（`.env` 文件）管理：

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `EVAL_LLM_*` | - | 评判模型（API Key、Base URL、模型名） |
| `SIM_LLM_*` | - | 用户模拟模型 |
| `AGENT_LLM_*` | - | 被测对话模型 |
| `ASSISTANT_LLM_API_KEY` | 留空回退到 `EVAL_LLM_API_KEY` | 内置 AI 助手 API Key（推荐独立低成本 key） |
| `ASSISTANT_LLM_BASE_URL` | 留空回退到 `EVAL_LLM_BASE_URL` | 内置 AI 助手 API 地址 |
| `ASSISTANT_LLM_MODEL` | 留空回退到 `EVAL_LLM_MODEL` | 内置 AI 助手模型名（推荐 gpt-4o-mini） |
| `MAX_TURNS` | 30 | 最大对话轮次 |
| `SIM_CONCURRENCY` | 8 | 并发对话模拟数 |
| `EVAL_CONCURRENCY` | 8 | 并发评测评判数 |
| `EVAL_BATCH_SIZE` | 12 | 每批 Persona 数量（断点续跑粒度） |
| `SELF_CONSISTENCY_RUNS` | 3 | 主观维度重复评判次数 |
| `LLM_RATE_LIMIT_RPS` | 0（禁用） | 每秒请求数限制 |
| `WEIGHT_FLOW` | 0.25 | 流程遵循维度权重 |
| `WEIGHT_CONSTRAINT` | 0.20 | 约束合规维度权重 |
| `WEIGHT_KNOWLEDGE` | 0.20 | 知识准确维度权重 |
| `WEIGHT_STYLE` | 0.10 | 风格语气维度权重 |
| `WEIGHT_COHERENCE` | 0.10 | 连贯性维度权重 |
| `WEIGHT_SAFETY` | 0.10 | 安全合规维度权重 |
| `WEIGHT_ADAPTABILITY` | 0.05 | 应变能力维度权重 |

维度权重会自动归一化，只需保持正确的相对比例即可。

## 部署

### Docker 部署

```bash
docker compose up -d
```

### 云服务器一键部署

编辑 `deploy.sh` 顶部的服务器配置，然后执行：

```bash
chmod +x deploy.sh
./deploy.sh
```

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。
