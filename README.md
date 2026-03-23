# DatabricksCheckpointSaver

LangGraph의 `BaseCheckpointSaver`를 **Delta Lake** 기반으로 구현한 Databricks 전용 메모리 세이버입니다.

에이전트의 대화 상태(체크포인트)를 Delta 테이블에 영속 저장하므로, 노트북 재시작이나 다른 세션에서도 이전 대화가 이어집니다.

---

## 목차 (한국어)

- [환경별 세이버 선택](#환경별-세이버-선택)
- [폴더 구조](#폴더-구조)
- [설치](#설치)
- [환경 1 — 노트북 / Databricks Job](#환경-1--노트북--databricks-job)
- [환경 2 — 모델 등록 (MLflow 배포 전)](#환경-2--모델-등록-mlflow-배포-전)
- [환경 3 — Model Serving 엔드포인트](#환경-3--model-serving-엔드포인트)
- [Delta 테이블 구조](#delta-테이블-구조)
- [API 레퍼런스](#api-레퍼런스)
- [멀티유저 패턴](#멀티유저-패턴)
- [시간여행](#시간여행)
- [주의사항](#주의사항)

---

## 환경별 세이버 선택

실행 환경에 따라 사용하는 세이버가 다릅니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   노트북 / Job                    모델 등록 노트북                │
│   (SparkSession 있음)             (SparkSession 있음)            │
│         │                               │                      │
│         ▼                               ▼                      │
│  DatabricksCheckpointSaver    DatabricksSQLCheckpointSaver      │
│  (Spark + Delta 직접 쓰기)     setup() 으로 테이블만 미리 생성   │
│                                         │                      │
│                                         ▼                      │
│                              Model Serving Endpoint            │
│                              (SparkSession 없음)               │
│                                         │                      │
│                                         ▼                      │
│                              DatabricksSQLCheckpointSaver      │
│                              (SQL Connector HTTP 연결)          │
└─────────────────────────────────────────────────────────────────┘
```

| 환경 | 세이버 | 이유 |
|------|--------|------|
| 노트북 / Databricks Job | `DatabricksCheckpointSaver` | SparkSession으로 Delta 직접 쓰기 |
| 모델 등록 노트북 (배포 준비) | `DatabricksSQLCheckpointSaver` | 테이블 사전 생성용 (`setup()`) |
| **Model Serving 엔드포인트** | `DatabricksSQLCheckpointSaver` | SparkSession 없음, SQL Connector로 연결 |

---

## 폴더 구조

```
DatabricksCheckpointSaver/
│
├── checkpointers/
│   ├── __init__.py                          # 두 세이버 모두 export
│   ├── databricks_checkpoint_saver.py       # Spark 기반 (노트북/Job)
│   └── databricks_sql_checkpoint_saver.py   # SQL Connector 기반 (Model Serving)
│
└── notebooks/
    ├── 01_setup_and_agent.py                # 노트북 환경 기본 예제
    ├── 02_multi_user_example.py             # 멀티유저/멀티세션 패턴
    └── 03_model_serving_example.py          # 모델 등록 → 배포 → 호출 전체 흐름
```

---

## 설치

### 노트북 / Job

```bash
%pip install langchain langgraph databricks-langchain deepagents
```

### 모델 등록 노트북 / Model Serving

```bash
%pip install langchain langgraph databricks-langchain databricks-sql-connector mlflow
```

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `langchain` | `>= 0.3.0` | `create_agent` (LangGraph v1 표준) |
| `langgraph` | `>= 1.0.0` | 그래프 실행 엔진 |
| `databricks-langchain` | `>= 0.1.0` | `ChatDatabricks` |
| `deepagents` | latest | `create_deep_agent` (플래닝/서브에이전트) |
| `databricks-sql-connector` | `>= 3.0.0` | Model Serving용 Delta 연결 |
| `pyspark` | Runtime 내장 | 노트북용 Delta 직접 쓰기 |

---

## 환경 1 — 노트북 / Databricks Job

`DatabricksCheckpointSaver`를 사용합니다. SparkSession이 자동으로 제공되므로 별도 생성이 불필요합니다.

### 세이버 초기화

```python
from checkpointers import DatabricksCheckpointSaver

# Databricks 노트북에서는 spark가 자동 주입됨
saver = DatabricksCheckpointSaver(
    spark=spark,
    catalog="main",
    schema="langgraph",
    table_prefix="langgraph",
)
saver.setup()  # 테이블 생성 (이미 있으면 스킵, 매 실행마다 호출 가능)
```

### 에이전트 생성

```python
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent   # LangGraph v1 (create_react_agent 대체)
from langchain_core.messages import HumanMessage

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=saver,
    system_prompt="당신은 친절한 어시스턴트입니다.",
)
```

복잡한 다단계 작업에는 `create_deep_agent`를 사용합니다:

```python
from deepagents import create_deep_agent

# 내장 도구: write_todos, read/write/edit_file, glob, grep, task(서브에이전트)
agent = create_deep_agent(model=llm, tools=[], checkpointer=saver)
```

### 대화 실행

```python
config = {"configurable": {"thread_id": "user-001"}}

result = agent.invoke(
    {"messages": [HumanMessage("안녕, 내 이름은 혜미야")]},
    config=config,
)
print(result["messages"][-1].content)

# 노트북을 재시작해도 이전 대화가 Delta에서 복원됨
result = agent.invoke(
    {"messages": [HumanMessage("내 이름이 뭐라고 했지?")]},
    config=config,
)
# → "혜미"
```

> 전체 예제: [`notebooks/01_setup_and_agent.py`](notebooks/01_setup_and_agent.py), [`notebooks/02_multi_user_example.py`](notebooks/02_multi_user_example.py)

---

## 환경 2 — 모델 등록 (MLflow 배포 전)

Model Serving 컨테이너에서는 `CREATE TABLE` 권한이 없을 수 있으므로,
**배포 전 노트북에서 `DatabricksSQLCheckpointSaver`로 테이블을 미리 생성**해야 합니다.

### 테이블 사전 생성

```python
import os
from checkpointers import DatabricksSQLCheckpointSaver

# 노트북 컨텍스트에서 인증 정보 로드
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"] = f"https://{ctx.browserHostName().get()}"
os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()

saver = DatabricksSQLCheckpointSaver(
    http_path="/sql/1.0/warehouses/<warehouse-id>",
    catalog="main",
    schema="langgraph",
    table_prefix="langgraph",
)
saver.setup()  # ← 배포 전 반드시 실행
print("✓ 테이블 생성 완료")
```

> `DATABRICKS_TOKEN`에 `w.config.token`은 노트북 환경에서 `None`을 반환할 수 있습니다.
> `dbutils`로 직접 컨텍스트에서 가져오는 것이 안전합니다.

### MLflow 모델 등록

```python
import mlflow

model_config = {
    "sql_warehouse_http_path": "/sql/1.0/warehouses/<warehouse-id>",
    "catalog": "main",
    "schema": "langgraph",
    "llm_endpoint": "databricks-meta-llama-3-3-70b-instruct",
}

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="langgraph_agent",
        python_model=LangGraphAgentModel(),
        model_config=model_config,
        code_paths=["/Workspace/Users/.../DatabricksCheckpointSaver/checkpointers"],
        pip_requirements=[
            "langchain>=0.3.0",
            "langgraph>=1.0.0",
            "databricks-langchain>=0.1.0",
            "databricks-sql-connector>=3.0.0",
        ],
        registered_model_name="main.langgraph.my_agent",
    )
```

> 전체 예제: [`notebooks/03_model_serving_example.py`](notebooks/03_model_serving_example.py)

---

## 환경 3 — Model Serving 엔드포인트

Model Serving 컨테이너는 **SparkSession이 없는** 경량 Python 환경입니다.
`DatabricksSQLCheckpointSaver`가 SQL Warehouse에 HTTP로 연결해 Delta 테이블을 읽고 씁니다.

### 아키텍처

```
[REST 클라이언트]
    │  POST /invocations  {"thread_id": "user-42", "messages": [...]}
    ▼
[Model Serving Endpoint]  ← SparkSession 없음
    │  ChatDatabricks  →  Foundation Model API
    │  DatabricksSQLCheckpointSaver
    │    └─ SQL Connector (HTTP)
    ▼
[SQL Warehouse]  ──▶  [Delta Lake: *_checkpoints 테이블]
```

### MLflow PythonModel 내부 구현

```python
class LangGraphAgentModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from databricks_langchain import ChatDatabricks
        from langchain.agents import create_agent
        from checkpointers import DatabricksSQLCheckpointSaver

        # DATABRICKS_HOST / DATABRICKS_TOKEN은 Model Serving이 자동 주입
        self.saver = DatabricksSQLCheckpointSaver(
            http_path=context.model_config["sql_warehouse_http_path"],
            catalog=context.model_config.get("catalog", "main"),
            schema=context.model_config.get("schema", "langgraph"),
        )

        llm = ChatDatabricks(
            endpoint=context.model_config.get("llm_endpoint"),
            temperature=0,
        )

        self.agent = create_agent(
            model=llm,
            tools=[],
            checkpointer=self.saver,
            system_prompt="당신은 친절한 어시스턴트입니다.",
        )

    def predict(self, context, model_input, params=None):
        import pandas as pd
        from langchain_core.messages import HumanMessage

        row = model_input.iloc[0].to_dict() if isinstance(model_input, pd.DataFrame) else model_input
        thread_id = row.get("thread_id", "default")
        last_msg = next(
            (m["content"] for m in reversed(row.get("messages", [])) if m["role"] == "user"), ""
        )

        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke({"messages": [HumanMessage(last_msg)]}, config=config)
        return {"thread_id": thread_id, "response": result["messages"][-1].content}
```

### 엔드포인트 호출

```python
import requests, os

response = requests.post(
    f"{os.environ['DATABRICKS_HOST']}/serving-endpoints/<endpoint-name>/invocations",
    headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
    json={
        "dataframe_split": {
            "columns": ["thread_id", "messages"],
            "data": [["user-001", [{"role": "user", "content": "안녕"}]]]
        }
    },
)
```

> 전체 예제: [`notebooks/03_model_serving_example.py`](notebooks/03_model_serving_example.py)

---

## Delta 테이블 구조

`setup()` 호출 시 두 테이블이 자동 생성됩니다.

### `{catalog}.{schema}.{prefix}_checkpoints`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `thread_id` | STRING | 사용자/세션 식별자 (파티션 키) |
| `checkpoint_ns` | STRING | 네임스페이스 (루트: `""`, 서브그래프: `"node:uuid"`) |
| `checkpoint_id` | STRING | 단조 증가하는 고유 체크포인트 ID |
| `parent_checkpoint_id` | STRING | 부모 체크포인트 ID (첫 체크포인트는 NULL) |
| `type` | STRING | 직렬화 타입 태그 |
| `checkpoint` | STRING | 직렬화된 체크포인트 데이터 |
| `metadata` | STRING | CheckpointMetadata (source, step, parents) |
| `created_at` | TIMESTAMP | 저장 시각 |

### `{catalog}.{schema}.{prefix}_checkpoint_writes`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `thread_id` | STRING | 사용자/세션 식별자 (파티션 키) |
| `checkpoint_ns` | STRING | 체크포인트 네임스페이스 |
| `checkpoint_id` | STRING | 이 쓰기가 속한 체크포인트 ID |
| `task_id` | STRING | 그래프 노드 태스크 ID |
| `idx` | INT | 태스크 내 쓰기 순서 |
| `channel` | STRING | 쓰여진 채널 이름 |
| `type` | STRING | 직렬화 타입 태그 |
| `value` | STRING | 직렬화된 채널 값 |

---

## API 레퍼런스

### `DatabricksCheckpointSaver`

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `spark` | 필수 | SparkSession 인스턴스 |
| `catalog` | `"main"` | Unity Catalog 카탈로그 |
| `schema` | `"langgraph"` | 스키마 이름 |
| `table_prefix` | `"langgraph"` | 테이블 이름 접두사 |

### `DatabricksSQLCheckpointSaver`

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `http_path` | 필수 | SQL Warehouse HTTP path |
| `catalog` | `"main"` | Unity Catalog 카탈로그 |
| `schema` | `"langgraph"` | 스키마 이름 |
| `table_prefix` | `"langgraph"` | 테이블 이름 접두사 |
| `server_hostname` | env `DATABRICKS_HOST` | 워크스페이스 호스트명 |
| `access_token` | env `DATABRICKS_TOKEN` | 액세스 토큰 |

### 공통 메서드

```python
saver.setup()                                       # 테이블 생성 (idempotent)
saver.get_tuple(config)                             # 최신 체크포인트 조회
saver.list(config, filter={"source": "loop"})       # 체크포인트 목록 (최신순)
saver.delete_thread("user-001")                     # 스레드 전체 삭제
saver.get_thread_history("user-001", limit=20)      # 히스토리 DataFrame (노트북 전용)
```

---

## 멀티유저 패턴

`thread_id`를 사용자/세션 ID로 사용하면 대화가 완전히 분리됩니다.

```python
def chat(user_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": user_id}}
    result = agent.invoke({"messages": [HumanMessage(message)]}, config=config)
    return result["messages"][-1].content

chat("alice", "안녕, 나는 Alice야")
chat("bob",   "안녕, 나는 Bob이야")
chat("alice", "내 이름이 뭐라고 했지?")  # → "Alice" (Bob 대화와 완전 분리)
```

---

## 시간여행

`list()`로 과거 체크포인트를 조회하고 특정 시점으로 되감을 수 있습니다.

```python
config = {"configurable": {"thread_id": "user-001"}}

checkpoints = list(saver.list(config))
for cp in checkpoints:
    print(f"step={cp.metadata.get('step'):<3}  id={cp.checkpoint['id']}")

# 특정 시점으로 되감아 재실행
replay_config = checkpoints[2].config
result = agent.invoke({"messages": [HumanMessage("여기서부터 다시")]}, config=replay_config)
```

---

## 주의사항

- `setup()`은 매 실행마다 호출해도 안전합니다 (`CREATE TABLE IF NOT EXISTS`).
- `DATABRICKS_TOKEN` 취득 시 `w.config.token`은 노트북 환경에서 `None`일 수 있습니다. `dbutils.notebook.entry_point`를 사용하세요.
- `DatabricksSQLCheckpointSaver`의 테이블은 Model Serving 배포 **전** 노트북에서 미리 생성해야 합니다.
- `create_react_agent`는 LangGraph v1에서 deprecated되었습니다. `langchain.agents.create_agent`를 사용하세요.

---
---

# English

---

## Table of Contents (English)

- [Choosing a Saver by Environment](#choosing-a-saver-by-environment)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Environment 1 — Notebook / Databricks Job](#environment-1--notebook--databricks-job)
- [Environment 2 — Model Registration (before MLflow deployment)](#environment-2--model-registration-before-mlflow-deployment)
- [Environment 3 — Model Serving Endpoint](#environment-3--model-serving-endpoint)
- [Delta Table Schema](#delta-table-schema)
- [API Reference](#api-reference)
- [Multi-user Pattern](#multi-user-pattern)
- [Time Travel](#time-travel)
- [Notes](#notes)

---

## Choosing a Saver by Environment

The correct saver depends on where your code runs.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Notebook / Job                  Model Registration Notebook  │
│   (SparkSession available)        (SparkSession available)     │
│         │                               │                      │
│         ▼                               ▼                      │
│  DatabricksCheckpointSaver    DatabricksSQLCheckpointSaver      │
│  (Spark + Delta direct write)  Run setup() to pre-create tables│
│                                         │                      │
│                                         ▼                      │
│                              Model Serving Endpoint            │
│                              (No SparkSession)                 │
│                                         │                      │
│                                         ▼                      │
│                              DatabricksSQLCheckpointSaver      │
│                              (SQL Connector over HTTP)         │
└─────────────────────────────────────────────────────────────────┘
```

| Environment | Saver | Reason |
|-------------|-------|--------|
| Notebook / Databricks Job | `DatabricksCheckpointSaver` | Writes to Delta directly via SparkSession |
| Model Registration Notebook | `DatabricksSQLCheckpointSaver` | Pre-create tables before serving (`setup()`) |
| **Model Serving Endpoint** | `DatabricksSQLCheckpointSaver` | No SparkSession available; connects via SQL Connector |

---

## Folder Structure

```
DatabricksCheckpointSaver/
│
├── checkpointers/
│   ├── __init__.py                          # Exports both savers
│   ├── databricks_checkpoint_saver.py       # Spark-based (Notebook / Job)
│   └── databricks_sql_checkpoint_saver.py   # SQL Connector-based (Model Serving)
│
└── notebooks/
    ├── 01_setup_and_agent.py                # Basic notebook example
    ├── 02_multi_user_example.py             # Multi-user / multi-session pattern
    └── 03_model_serving_example.py          # Full flow: register → deploy → invoke
```

---

## Installation

### Notebook / Job

```bash
%pip install langchain langgraph databricks-langchain deepagents
```

### Model Registration / Model Serving

```bash
%pip install langchain langgraph databricks-langchain databricks-sql-connector mlflow
```

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | `>= 0.3.0` | `create_agent` (LangGraph v1 standard) |
| `langgraph` | `>= 1.0.0` | Graph execution engine |
| `databricks-langchain` | `>= 0.1.0` | `ChatDatabricks` LLM |
| `deepagents` | latest | `create_deep_agent` (planning / subagents) |
| `databricks-sql-connector` | `>= 3.0.0` | Delta access for Model Serving |
| `pyspark` | Runtime built-in | Delta direct write for notebooks |

---

## Environment 1 — Notebook / Databricks Job

Use `DatabricksCheckpointSaver`. The `spark` variable is automatically injected in Databricks notebooks.

### Initialize the saver

```python
from checkpointers import DatabricksCheckpointSaver

# spark is auto-injected in Databricks notebooks — no need to call getOrCreate()
saver = DatabricksCheckpointSaver(
    spark=spark,
    catalog="main",
    schema="langgraph",
    table_prefix="langgraph",
)
saver.setup()  # Creates tables if not exist — safe to call every run
```

### Create the agent

```python
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent   # LangGraph v1 (replaces create_react_agent)
from langchain_core.messages import HumanMessage

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=saver,
    system_prompt="You are a helpful assistant.",
)
```

For complex multi-step tasks, use `create_deep_agent`:

```python
from deepagents import create_deep_agent

# Built-in tools: write_todos, read/write/edit_file, glob, grep, task (subagent)
agent = create_deep_agent(model=llm, tools=[], checkpointer=saver)
```

### Run a conversation

```python
config = {"configurable": {"thread_id": "user-001"}}

result = agent.invoke({"messages": [HumanMessage("Hi, my name is Hyemi")]}, config=config)
print(result["messages"][-1].content)

# After notebook restart, history is restored from Delta
result = agent.invoke({"messages": [HumanMessage("What's my name?")]}, config=config)
# → "Hyemi"
```

> Full example: [`notebooks/01_setup_and_agent.py`](notebooks/01_setup_and_agent.py), [`notebooks/02_multi_user_example.py`](notebooks/02_multi_user_example.py)

---

## Environment 2 — Model Registration (before MLflow deployment)

The Model Serving container may not have `CREATE TABLE` permission.
**Run `setup()` from a notebook before deploying**, using `DatabricksSQLCheckpointSaver`.

### Pre-create tables

```python
import os
from checkpointers import DatabricksSQLCheckpointSaver

# Load credentials from notebook context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"] = f"https://{ctx.browserHostName().get()}"
os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()

saver = DatabricksSQLCheckpointSaver(
    http_path="/sql/1.0/warehouses/<warehouse-id>",
    catalog="main",
    schema="langgraph",
    table_prefix="langgraph",
)
saver.setup()  # ← Must run before deploying to Model Serving
print("✓ Tables created")
```

> `w.config.token` may return `None` in notebook environments.
> Use `dbutils` to retrieve credentials from the notebook context instead.

### Register the MLflow model

```python
import mlflow

model_config = {
    "sql_warehouse_http_path": "/sql/1.0/warehouses/<warehouse-id>",
    "catalog": "main",
    "schema": "langgraph",
    "llm_endpoint": "databricks-meta-llama-3-3-70b-instruct",
}

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="langgraph_agent",
        python_model=LangGraphAgentModel(),
        model_config=model_config,
        code_paths=["/Workspace/Users/.../DatabricksCheckpointSaver/checkpointers"],
        pip_requirements=[
            "langchain>=0.3.0",
            "langgraph>=1.0.0",
            "databricks-langchain>=0.1.0",
            "databricks-sql-connector>=3.0.0",
        ],
        registered_model_name="main.langgraph.my_agent",
    )
```

> Full example: [`notebooks/03_model_serving_example.py`](notebooks/03_model_serving_example.py)

---

## Environment 3 — Model Serving Endpoint

Model Serving containers run in a **lightweight Python environment with no SparkSession**.
`DatabricksSQLCheckpointSaver` connects to a SQL Warehouse over HTTP to read/write Delta tables.

### Architecture

```
[REST Client]
    │  POST /invocations  {"thread_id": "user-42", "messages": [...]}
    ▼
[Model Serving Endpoint]  ← No SparkSession
    │  ChatDatabricks  →  Foundation Model API
    │  DatabricksSQLCheckpointSaver
    │    └─ SQL Connector (HTTP)
    ▼
[SQL Warehouse]  ──▶  [Delta Lake: *_checkpoints table]
```

### MLflow PythonModel implementation

```python
class LangGraphAgentModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from databricks_langchain import ChatDatabricks
        from langchain.agents import create_agent
        from checkpointers import DatabricksSQLCheckpointSaver

        # DATABRICKS_HOST and DATABRICKS_TOKEN are auto-injected by Model Serving
        self.saver = DatabricksSQLCheckpointSaver(
            http_path=context.model_config["sql_warehouse_http_path"],
            catalog=context.model_config.get("catalog", "main"),
            schema=context.model_config.get("schema", "langgraph"),
        )

        llm = ChatDatabricks(
            endpoint=context.model_config.get("llm_endpoint"),
            temperature=0,
        )

        self.agent = create_agent(
            model=llm,
            tools=[],
            checkpointer=self.saver,
            system_prompt="You are a helpful assistant.",
        )

    def predict(self, context, model_input, params=None):
        import pandas as pd
        from langchain_core.messages import HumanMessage

        row = model_input.iloc[0].to_dict() if isinstance(model_input, pd.DataFrame) else model_input
        thread_id = row.get("thread_id", "default")
        last_msg = next(
            (m["content"] for m in reversed(row.get("messages", [])) if m["role"] == "user"), ""
        )

        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke({"messages": [HumanMessage(last_msg)]}, config=config)
        return {"thread_id": thread_id, "response": result["messages"][-1].content}
```

### Invoke the endpoint

```python
import requests, os

response = requests.post(
    f"{os.environ['DATABRICKS_HOST']}/serving-endpoints/<endpoint-name>/invocations",
    headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
    json={
        "dataframe_split": {
            "columns": ["thread_id", "messages"],
            "data": [["user-001", [{"role": "user", "content": "Hello"}]]]
        }
    },
)
```

> Full example: [`notebooks/03_model_serving_example.py`](notebooks/03_model_serving_example.py)

---

## Delta Table Schema

Both tables are created by `setup()`.

### `{catalog}.{schema}.{prefix}_checkpoints`

| Column | Type | Description |
|--------|------|-------------|
| `thread_id` | STRING | User/session identifier (partition key) |
| `checkpoint_ns` | STRING | Namespace (`""` for root, `"node:uuid"` for subgraphs) |
| `checkpoint_id` | STRING | Monotonically increasing unique checkpoint ID |
| `parent_checkpoint_id` | STRING | Parent checkpoint ID (NULL for first checkpoint) |
| `type` | STRING | Serializer type tag |
| `checkpoint` | STRING | Serialized checkpoint data |
| `metadata` | STRING | CheckpointMetadata (source, step, parents) |
| `created_at` | TIMESTAMP | Write timestamp |

### `{catalog}.{schema}.{prefix}_checkpoint_writes`

| Column | Type | Description |
|--------|------|-------------|
| `thread_id` | STRING | User/session identifier (partition key) |
| `checkpoint_ns` | STRING | Checkpoint namespace |
| `checkpoint_id` | STRING | Checkpoint this write belongs to |
| `task_id` | STRING | Graph node task ID |
| `idx` | INT | Write index within the task |
| `channel` | STRING | Channel name written to |
| `type` | STRING | Serializer type tag |
| `value` | STRING | Serialized channel value |

---

## API Reference

### `DatabricksCheckpointSaver`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spark` | required | SparkSession instance |
| `catalog` | `"main"` | Unity Catalog catalog name |
| `schema` | `"langgraph"` | Schema name |
| `table_prefix` | `"langgraph"` | Table name prefix |

### `DatabricksSQLCheckpointSaver`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `http_path` | required | SQL Warehouse HTTP path |
| `catalog` | `"main"` | Unity Catalog catalog name |
| `schema` | `"langgraph"` | Schema name |
| `table_prefix` | `"langgraph"` | Table name prefix |
| `server_hostname` | env `DATABRICKS_HOST` | Workspace hostname |
| `access_token` | env `DATABRICKS_TOKEN` | Access token |

### Shared methods

```python
saver.setup()                                       # Create tables (idempotent)
saver.get_tuple(config)                             # Get latest checkpoint
saver.list(config, filter={"source": "loop"})       # List checkpoints (newest first)
saver.delete_thread("user-001")                     # Delete all checkpoints for a thread
saver.get_thread_history("user-001", limit=20)      # History DataFrame (notebook only)
```

---

## Multi-user Pattern

Use `thread_id` as a user or session identifier to keep conversations fully isolated.

```python
def chat(user_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": user_id}}
    result = agent.invoke({"messages": [HumanMessage(message)]}, config=config)
    return result["messages"][-1].content

chat("alice", "Hi, I'm Alice")
chat("bob",   "Hi, I'm Bob")
chat("alice", "What's my name?")  # → "Alice" (isolated from Bob's conversation)
```

---

## Time Travel

Use `list()` to browse checkpoint history and replay from any past state.

```python
config = {"configurable": {"thread_id": "user-001"}}

checkpoints = list(saver.list(config))
for cp in checkpoints:
    print(f"step={cp.metadata.get('step'):<3}  id={cp.checkpoint['id']}")

# Rewind to a past checkpoint and reinvoke
replay_config = checkpoints[2].config
result = agent.invoke({"messages": [HumanMessage("Start over from here")]}, config=replay_config)
```

---

## Notes

- `setup()` is safe to call on every run (`CREATE TABLE IF NOT EXISTS`).
- `w.config.token` may return `None` in notebook environments. Use `dbutils.notebook.entry_point` to retrieve the token from the notebook context.
- Delta tables must be pre-created via `setup()` in a notebook **before** deploying to Model Serving.
- `create_react_agent` is deprecated in LangGraph v1. Use `langchain.agents.create_agent` instead.
