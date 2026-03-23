# DatabricksCheckpointSaver

LangGraph의 `BaseCheckpointSaver`를 **Delta Lake** 기반으로 구현한 Databricks 전용 메모리 세이버입니다.

에이전트의 대화 상태(체크포인트)를 Delta 테이블에 영속 저장하므로, 노트북 재시작이나 다른 세션에서도 이전 대화가 이어집니다.

---

## 목차

- [특징](#특징)
- [폴더 구조](#폴더-구조)
- [설치 및 요구사항](#설치-및-요구사항)
- [빠른 시작](#빠른-시작)
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
│   ├── __init__.py                      # DatabricksCheckpointSaver export
│   └── databricks_checkpoint_saver.py   # 핵심 구현체
│
└── notebooks/
    ├── 01_setup_and_agent.py            # 기본 설정 및 React Agent 예제
    └── 02_multi_user_example.py         # 멀티유저/멀티세션 패턴
```

### 각 파일 역할

**`checkpointers/databricks_checkpoint_saver.py`**
- `DatabricksCheckpointSaver` 클래스 정의
- `BaseCheckpointSaver` 추상 클래스 구현 (`get_tuple`, `list`, `put`, `put_writes`)
- 동기/비동기 인터페이스 모두 구현

**`notebooks/01_setup_and_agent.py`**
- 테이블 초기화 (`setup()`)
- React Agent 생성 및 대화 예제
- 체크포인트 히스토리 조회 예제

**`notebooks/02_multi_user_example.py`**
- 여러 사용자를 `thread_id`로 분리하는 패턴
- Delta 테이블에서 유저별 활동 집계
- 대화 전체 복원 예제

---

## 설치 및 요구사항

### Databricks 클러스터 / 노트북

```bash
%pip install langgraph langchain-openai langchain-core
```

| 패키지 | 버전 |
|--------|------|
| `langgraph` | `>= 0.2.0` |
| `langchain-core` | `>= 0.2.0` |
| `pyspark` | Databricks Runtime 내장 |

### 권한

- Unity Catalog에서 `CREATE SCHEMA`, `CREATE TABLE` 권한 필요
- 또는 이미 존재하는 스키마에 `CREATE TABLE` 권한

---

## 빠른 시작

### 1. 레포지토리를 Databricks Repos에 추가

Databricks 워크스페이스 → **Repos** → **Add Repo** → 이 레포 URL 입력

### 2. 노트북에서 import

```python
import sys
sys.path.insert(0, "/Workspace/Repos/<your-username>/DatabricksCheckpointSaver")

from checkpointers import DatabricksCheckpointSaver
```

### 3. 세이버 초기화 및 테이블 생성

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

### 4. Agent에 주입

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[], checkpointer=saver)
```

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
- 비동기(`async`) 인터페이스도 구현되어 있으나, Databricks 클러스터 환경에서는 동기 방식이 권장됩니다.
- 체크포인트 데이터는 `JsonPlusSerializer`로 직렬화됩니다. LangGraph 버전 업그레이드 시 직렬화 포맷 변경에 주의하세요.
