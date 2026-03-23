# DatabricksCheckpointSaver

LangGraph의 `BaseCheckpointSaver`를 **Delta Lake** 기반으로 구현한 Databricks 전용 메모리 세이버입니다.

에이전트의 대화 상태(체크포인트)를 Delta 테이블에 영속 저장하므로, 노트북 재시작이나 다른 세션에서도 이전 대화가 이어집니다.

**실행 환경에 따라 두 가지 세이버를 제공합니다:**

| 세이버 | 환경 | 방식 |
|--------|------|------|
| `DatabricksCheckpointSaver` | 노트북 / Databricks Job | SparkSession + Delta 직접 쓰기 |
| `DatabricksSQLCheckpointSaver` | **Model Serving** / 로컬 | SQL Connector (HTTP) — SparkSession 불필요 |

---

## 목차

- [특징](#특징)
- [폴더 구조](#폴더-구조)
- [설치 및 요구사항](#설치-및-요구사항)
- [빠른 시작](#빠른-시작)
- [Model Serving 배포](#model-serving-배포)
- [Delta 테이블 구조](#delta-테이블-구조)
- [API 레퍼런스](#api-레퍼런스)
- [멀티유저 패턴](#멀티유저-패턴)
- [대화 히스토리 조회](#대화-히스토리-조회)
- [시간여행 (Time Travel)](#시간여행-time-travel)
- [주의사항](#주의사항)

---

## 특징

| 기능 | 설명 |
|------|------|
| **영속 메모리** | Delta Lake에 저장 → 노트북/클러스터 재시작 후에도 대화 유지 |
| **멀티유저 지원** | `thread_id`로 사용자/세션별 대화 완전 분리 |
| **Idempotent 쓰기** | `MERGE INTO` 사용 → 동일 체크포인트 중복 저장 방지 |
| **SQL 안전성** | DataFrame API로 읽기, `createDataFrame` + MERGE로 쓰기 → SQL injection 없음 |
| **시간여행** | `list()`로 과거 체크포인트 조회 및 특정 시점으로 되감기 |
| **SQL 조회 가능** | Delta 테이블이므로 Spark SQL / DatabricksSQL로 직접 분석 가능 |
| **Unity Catalog 지원** | `catalog.schema.table` 3-레벨 네임스페이스 지원 |

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
    ├── 01_setup_and_agent.py                # 기본 설정 및 React Agent 예제
    ├── 02_multi_user_example.py             # 멀티유저/멀티세션 패턴
    └── 03_model_serving_example.py          # Model Serving 배포 전체 흐름
```

### 각 파일 역할

**`checkpointers/databricks_checkpoint_saver.py`**
- SparkSession + Delta Lake 직접 쓰기
- Databricks 노트북 / Job 환경에서 사용
- `createDataFrame` + `MERGE INTO`로 SQL injection 없이 안전하게 저장

**`checkpointers/databricks_sql_checkpoint_saver.py`**
- Databricks SQL Connector(HTTP)로 SQL Warehouse에 연결
- SparkSession 없이 동작 → **Model Serving 엔드포인트에서 사용**
- `threading.local()`로 스레드별 커넥션 관리
- 파라미터화된 쿼리(`%s`)로 SQL injection 방지
- `asyncio.to_thread`로 비동기 인터페이스 제공

**`notebooks/01_setup_and_agent.py`**
- 테이블 초기화, React Agent 생성, 대화 예제

**`notebooks/02_multi_user_example.py`**
- `thread_id`로 사용자를 분리하는 패턴, Delta 집계 쿼리

**`notebooks/03_model_serving_example.py`**
- MLflow PythonModel로 에이전트 패키징
- Model Serving 엔드포인트 생성 및 REST 호출 예제

---

## 설치 및 요구사항

### 노트북 / Job 환경 (`DatabricksCheckpointSaver`)

```bash
%pip install langchain langgraph databricks-langchain deepagents
```

### Model Serving 환경 (`DatabricksSQLCheckpointSaver`)

```bash
pip install langchain langgraph databricks-langchain databricks-sql-connector mlflow
```

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `langchain` | `>= 0.3.0` | `create_agent` (LangGraph v1 표준) |
| `langgraph` | `>= 1.0.0` | 그래프 실행 엔진 |
| `databricks-langchain` | `>= 0.1.0` | `ChatDatabricks`, `DatabricksEmbeddings` |
| `deepagents` | latest | `create_deep_agent` (플래닝/서브에이전트) |
| `databricks-sql-connector` | `>= 3.0.0` | Model Serving용 Delta 연결 |
| `pyspark` | Runtime 내장 | 노트북용 Delta 직접 쓰기 |

### 권한

- Unity Catalog에서 `CREATE SCHEMA`, `CREATE TABLE` 권한 필요
- Model Serving 환경에서는 `DATABRICKS_HOST`, `DATABRICKS_TOKEN`이 자동 주입됩니다

---

## 빠른 시작

### 1. 레포지토리를 Databricks Repos에 추가

Databricks 워크스페이스 → **Repos** → **Add Repo** → 이 레포 URL 입력

같은 Repo 안의 노트북에서는 `sys.path` 조작 없이 바로 import 됩니다.

```python
# Repos 내 노트북이면 이것만으로 충분
from checkpointers import DatabricksCheckpointSaver
```

### 2. 세이버 초기화 및 테이블 생성

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

saver = DatabricksCheckpointSaver(
    spark=spark,
    catalog="main",       # Unity Catalog 카탈로그 이름
    schema="langgraph",   # 스키마 이름 (없으면 자동 생성)
    table_prefix="langgraph",  # 테이블 이름 접두사
)
saver.setup()  # 테이블 생성 (이미 있으면 스킵 — 매 실행마다 호출 가능)
```

### 4. LLM 선택 — `ChatDatabricks` 권장

`databricks-langchain`의 `ChatDatabricks`를 사용하면 OpenAI API 키 없이
워크스페이스 인증만으로 Databricks Foundation Model API를 호출할 수 있습니다.

```python
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent   # LangGraph v1 표준 (create_react_agent 대체)

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

# create_agent: prompt= → system_prompt= 로 파라미터명 변경됨
agent = create_agent(model=llm, tools=[], checkpointer=saver, system_prompt="...")
```

복잡한 다단계 작업이라면 `create_deep_agent`를 사용합니다:

```python
from deepagents import create_deep_agent

# 내장 도구: write_todos, read/write/edit_file, glob, grep, task(서브에이전트)
deep_agent = create_deep_agent(model=llm, tools=[], checkpointer=saver)
```

주요 Foundation Model 엔드포인트:

| 엔드포인트 | 모델 |
|------------|------|
| `databricks-meta-llama-3-3-70b-instruct` | Llama 3.3 70B |
| `databricks-claude-sonnet-4` | Claude Sonnet 4 |
| `databricks-dbrx-instruct` | DBRX |

> 워크스페이스 → **Serving** → **Foundation Model APIs** 에서 사용 가능한 엔드포인트를 확인하세요.

### 5. 대화 실행 — thread_id로 세션 구분

```python
# thread_id가 같으면 → 이전 대화가 Delta에서 자동 복원
config = {"configurable": {"thread_id": "user-001"}}

result = agent.invoke(
    {"messages": [HumanMessage("안녕, 내 이름은 혜미야")]},
    config=config,
)
print(result["messages"][-1].content)

# 노트북을 재시작해도 아래 대화가 이어짐
result = agent.invoke(
    {"messages": [HumanMessage("내 이름이 뭐라고 했지?")]},
    config=config,
)
print(result["messages"][-1].content)
# → "혜미"라고 답해야 정상
```

---

## Delta 테이블 구조

`setup()` 호출 시 아래 두 테이블이 자동 생성됩니다.

### `{catalog}.{schema}.{prefix}_checkpoints`

대화 상태의 전체 스냅샷 (체크포인트) 저장.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `thread_id` | STRING | 사용자/세션 식별자 (파티션 키) |
| `checkpoint_ns` | STRING | 네임스페이스 (루트: `""`, 서브그래프: `"node:uuid"`) |
| `checkpoint_id` | STRING | 단조 증가하는 고유 체크포인트 ID |
| `parent_checkpoint_id` | STRING | 부모 체크포인트 ID (첫 체크포인트는 NULL) |
| `type` | STRING | 직렬화 타입 태그 |
| `checkpoint` | STRING | 직렬화된 체크포인트 JSON |
| `metadata` | STRING | `CheckpointMetadata` JSON (source, step, parents) |
| `created_at` | TIMESTAMP | 저장 시각 |

### `{catalog}.{schema}.{prefix}_checkpoint_writes`

그래프 노드의 중간 쓰기(Pending Writes) 저장. 노드 실패 시 복구에 사용.

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

### `DatabricksCheckpointSaver(spark, catalog, schema, table_prefix)`

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `spark` | 필수 | `SparkSession` 인스턴스 |
| `catalog` | `"main"` | Unity Catalog 카탈로그 이름 |
| `schema` | `"langgraph"` | 스키마 이름 |
| `table_prefix` | `"langgraph"` | 테이블 이름 접두사 |

### 주요 메서드

```python
# 테이블 생성 (idempotent)
saver.setup()

# 특정 스레드의 최신 체크포인트 조회
saver.get_tuple(config)

# 체크포인트 히스토리 이터레이터 (최신순)
saver.list(config, filter={"source": "loop"}, limit=10)

# 스레드 전체 삭제
saver.delete_thread("user-001")

# 히스토리 DataFrame 반환 (디버깅/모니터링용)
saver.get_thread_history("user-001", limit=20)
```

---

## Model Serving 배포

Model Serving 환경에는 SparkSession이 없으므로 `DatabricksSQLCheckpointSaver`를 사용합니다.

### 아키텍처

```
[REST 클라이언트]
    │  POST /invocations  {"thread_id": "user-42", "messages": [...]}
    ▼
[Model Serving Endpoint]
    │  DatabricksSQLCheckpointSaver
    │  (DATABRICKS_HOST / DATABRICKS_TOKEN 자동 주입)
    ▼
[SQL Warehouse]  ──▶  [Delta Lake: langgraph_checkpoints]
```

### 사용법

```python
# Model Serving 엔드포인트 코드 (mlflow.pyfunc.PythonModel 내부)
from checkpointers import DatabricksSQLCheckpointSaver

class MyAgentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.saver = DatabricksSQLCheckpointSaver(
            http_path=context.model_config["sql_warehouse_http_path"],
            # server_hostname, access_token은 env var에서 자동 로드
        )
        self.agent = create_react_agent(llm, tools, checkpointer=self.saver)

    def predict(self, context, model_input, params=None):
        config = {"configurable": {"thread_id": model_input["thread_id"]}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(model_input["message"])]},
            config=config,
        )
        return result["messages"][-1].content
```

> **배포 전 주의:** `setup()`으로 테이블을 미리 생성해두세요.
> Model Serving 컨테이너에는 `CREATE TABLE` 권한이 없을 수 있습니다.

전체 예제는 [`notebooks/03_model_serving_example.py`](notebooks/03_model_serving_example.py)를 참고하세요.

---

## 멀티유저 패턴

`thread_id`를 사용자 ID나 세션 ID로 사용하면 대화가 완전히 분리됩니다.

```python
def chat(user_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": user_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(message)]},
        config=config,
    )
    return result["messages"][-1].content

# 각 유저의 대화는 독립적으로 저장/복원
chat("alice", "안녕, 나는 Alice야")
chat("bob",   "안녕, 나는 Bob이야")
chat("alice", "내 이름이 뭐라고 했지?")  # → "Alice" (Bob 대화와 완전 분리)
```

---

## 대화 히스토리 조회

Delta 테이블이므로 Spark SQL로 직접 분석할 수 있습니다.

```python
from pyspark.sql import functions as F

# 유저별 체크포인트 수 / 마지막 활동 시간
spark.table("main.langgraph.langgraph_checkpoints") \
    .groupBy("thread_id") \
    .agg(
        F.count("*").alias("checkpoint_count"),
        F.max("created_at").alias("last_activity"),
    ) \
    .orderBy(F.col("last_activity").desc()) \
    .display()

# 특정 유저의 최근 20개 체크포인트
saver.get_thread_history("alice", limit=20).display()
```

---

## 시간여행 (Time Travel)

`list()`로 과거 체크포인트를 조회하고, 특정 시점으로 되감아 재실행할 수 있습니다.

```python
config = {"configurable": {"thread_id": "user-001"}}

# 전체 체크포인트 목록 (최신순)
checkpoints = list(saver.list(config))
for cp in checkpoints:
    print(f"step={cp.metadata.get('step'):<3}  id={cp.checkpoint['id']}")

# 3번째 이전 체크포인트로 되감아 재실행
replay_config = checkpoints[2].config
result = agent.invoke(
    {"messages": [HumanMessage("여기서부터 다시 시작")]},
    config=replay_config,
)
```

---

## 주의사항

- `setup()`은 매 노트북 실행 시 호출해도 안전합니다 (`CREATE TABLE IF NOT EXISTS` 사용).
- `thread_id`는 문자열이면 무엇이든 사용 가능합니다 (UUID, 유저 이름, 세션 ID 등).
- `DatabricksSQLCheckpointSaver`는 Model Serving 배포 **전** 노트북에서 `setup()`을 실행해 테이블을 미리 만들어두세요.
- 비동기(`async`) 인터페이스도 구현되어 있으나, Databricks 클러스터 환경에서는 동기 방식이 권장됩니다.
- 체크포인트 데이터는 `JsonPlusSerializer`로 직렬화됩니다. LangGraph 버전 업그레이드 시 직렬화 포맷 변경에 주의하세요.
