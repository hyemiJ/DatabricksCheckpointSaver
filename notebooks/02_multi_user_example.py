# Databricks notebook source
# MAGIC %md
# MAGIC # 멀티 유저 / 멀티 에이전트 패턴
# MAGIC
# MAGIC `thread_id`를 유저 ID나 세션 ID로 사용하면 여러 사용자의 대화를 독립적으로 관리할 수 있습니다.

# COMMAND ----------

from pyspark.sql import SparkSession
from checkpointers import DatabricksCheckpointSaver
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

spark = SparkSession.builder.getOrCreate()

saver = DatabricksCheckpointSaver(spark, catalog="main", schema="langgraph")
saver.setup()

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

@tool
def get_user_info(user_id: str) -> str:
    """사용자 정보를 조회합니다."""
    users = {"alice": "Alice (마케팅팀)", "bob": "Bob (개발팀)"}
    return users.get(user_id, "알 수 없는 사용자")

agent = create_agent(
    model=llm,
    tools=[get_user_info],
    checkpointer=saver,
    system_prompt="당신은 친절한 어시스턴트입니다.",
)

# COMMAND ----------

# MAGIC %md ## 사용자별 독립 대화

# COMMAND ----------

def chat(thread_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return result["messages"][-1].content

# Alice의 대화
print("=== Alice ===")
print(chat("alice-session-1", "안녕, 내 이름은 Alice야"))
print(chat("alice-session-1", "내 이름이 뭐라고 했지?"))  # "Alice"를 기억해야 함

print("\n=== Bob ===")
print(chat("bob-session-1", "안녕, 나는 Bob이야"))
print(chat("bob-session-1", "내 이름이 뭐지?"))           # "Bob"을 기억해야 함

# COMMAND ----------

# MAGIC %md ## Delta 테이블에서 유저별 체크포인트 수 집계

# COMMAND ----------

from pyspark.sql import functions as F

summary = (
    spark.table("main.langgraph.langgraph_checkpoints")
    .groupBy("thread_id")
    .agg(
        F.count("*").alias("checkpoint_count"),
        F.max("created_at").alias("last_activity"),
    )
    .orderBy(F.col("last_activity").desc())
)
display(summary)

# COMMAND ----------

# MAGIC %md ## 특정 유저의 전체 대화 복원

# COMMAND ----------

def replay_conversation(thread_id: str):
    """thread_id의 전체 대화 내역을 최신 체크포인트에서 복원합니다."""
    config = {"configurable": {"thread_id": thread_id}}
    cp_tuple = saver.get_tuple(config)
    if cp_tuple is None:
        print(f"No history for thread_id: {thread_id}")
        return

    messages = cp_tuple.checkpoint.get("channel_values", {}).get("messages", [])
    print(f"=== {thread_id} 대화 내역 ({len(messages)}개 메시지) ===")
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        print(f"[{role}] {content[:100]}")

replay_conversation("alice-session-1")
