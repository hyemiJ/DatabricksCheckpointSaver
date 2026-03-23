# Databricks notebook source
# MAGIC %md
# MAGIC # LangGraph Agent with Databricks Delta Lake Memory
# MAGIC
# MAGIC `DatabricksCheckpointSaver`는 LangGraph의 `BaseCheckpointSaver`를 Delta Lake 테이블로 구현합니다.
# MAGIC 대화 상태(체크포인트)가 Delta 테이블에 영속 저장되므로, 노트북을 재시작하거나 다른 세션에서도 이전 대화가 이어집니다.
# MAGIC
# MAGIC ## 구조
# MAGIC ```
# MAGIC main.langgraph.langgraph_checkpoints       — 체크포인트 (대화 스냅샷)
# MAGIC main.langgraph.langgraph_checkpoint_writes — 노드별 중간 쓰기
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install langgraph langchain-openai --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/<your-repo>/databricks")  # 본인 경로로 변경

from pyspark.sql import SparkSession
from checkpointers import DatabricksCheckpointSaver

spark = SparkSession.builder.getOrCreate()

# ── 1. 세이버 초기화 ──────────────────────────────────────────────────────────
saver = DatabricksCheckpointSaver(
    spark=spark,
    catalog="main",       # Unity Catalog 카탈로그
    schema="langgraph",   # 스키마 (자동 생성됨)
    table_prefix="langgraph",
)
saver.setup()  # 테이블 생성 (이미 있으면 스킵)
print("✓ 테이블 준비 완료")

# COMMAND ----------

# MAGIC %md ## 2. React Agent 생성

# COMMAND ----------

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import os

# Databricks Secret을 사용하거나 환경변수로 설정
# os.environ["OPENAI_API_KEY"] = dbutils.secrets.get("my-scope", "openai-api-key")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 도구 예시 — 실제 사용 시 원하는 도구로 교체
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}의 현재 날씨: 맑음, 22°C"

tools = [get_weather]

# checkpointer에 DatabricksCheckpointSaver 주입
agent = create_react_agent(llm, tools, checkpointer=saver)

# COMMAND ----------

# MAGIC %md ## 3. 대화 실행 — thread_id로 세션 구분

# COMMAND ----------

# thread_id = 사용자/세션 식별자. 같은 ID를 쓰면 대화가 이어집니다.
config = {"configurable": {"thread_id": "user-demo-001"}}

# 첫 번째 메시지
result = agent.invoke(
    {"messages": [HumanMessage(content="서울 날씨 알려줘")]},
    config=config,
)
print(result["messages"][-1].content)

# COMMAND ----------

# 같은 thread_id → 이전 대화 맥락이 Delta에서 복원됨
result = agent.invoke(
    {"messages": [HumanMessage(content="방금 알려준 도시가 어디야?")]},
    config=config,
)
print(result["messages"][-1].content)
# → "서울"이라고 대답해야 정상 (이전 대화를 기억)

# COMMAND ----------

# MAGIC %md ## 4. 체크포인트 히스토리 확인

# COMMAND ----------

# Delta 테이블에 저장된 체크포인트 히스토리 조회
history_df = saver.get_thread_history("user-demo-001")
display(history_df)

# COMMAND ----------

# 전체 테이블 조회
display(spark.table("main.langgraph.langgraph_checkpoints"))

# COMMAND ----------

# MAGIC %md ## 5. 특정 시점으로 되감기 (Time Travel)

# COMMAND ----------

# list()로 과거 체크포인트 목록 가져오기
checkpoints = list(saver.list(config))
for cp in checkpoints:
    print(f"step={cp.metadata.get('step'):<3}  id={cp.checkpoint['id']}")

# COMMAND ----------

# 특정 checkpoint_id로 되감아 재실행
if len(checkpoints) >= 2:
    replay_config = checkpoints[-2].config  # 두 번째로 오래된 체크포인트
    result = agent.invoke(
        {"messages": [HumanMessage(content="다시 처음부터 물어볼게, 서울 날씨?")]},
        config=replay_config,
    )
    print(result["messages"][-1].content)

# COMMAND ----------

# MAGIC %md ## 6. 스레드 삭제 (선택)

# COMMAND ----------

# saver.delete_thread("user-demo-001")
# print("스레드 삭제 완료")
