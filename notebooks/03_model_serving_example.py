# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving 엔드포인트에 LangGraph Agent 배포
# MAGIC
# MAGIC `DatabricksSQLCheckpointSaver` + `ChatDatabricks`를 사용합니다.
# MAGIC - **외부 API 키 불필요** — Databricks Foundation Model API 사용
# MAGIC - SparkSession 없이 SQL Connector로 Delta 메모리 유지
# MAGIC
# MAGIC ## 흐름
# MAGIC ```
# MAGIC [클라이언트]
# MAGIC     ↓  POST /invocations  {"thread_id": "user-42", "messages": [...]}
# MAGIC [Model Serving Endpoint]
# MAGIC     │  ChatDatabricks → Foundation Model API (워크스페이스 내)
# MAGIC     │  DatabricksSQLCheckpointSaver → SQL Warehouse → Delta Lake
# MAGIC     ↓
# MAGIC [응답 반환]
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install langchain langgraph databricks-langchain databricks-sql-connector mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Step 1 — 테이블 미리 생성 (배포 전 한 번만)
# MAGIC
# MAGIC Model Serving 컨테이너에서 `CREATE TABLE` 권한이 없을 수 있으므로
# MAGIC 노트북에서 미리 실행해둡니다.

# COMMAND ----------

from checkpointers import DatabricksSQLCheckpointSaver

SQL_WAREHOUSE_HTTP_PATH = "/sql/1.0/warehouses/<your-warehouse-id>"  # 변경 필요

saver = DatabricksSQLCheckpointSaver(
    http_path=SQL_WAREHOUSE_HTTP_PATH,
    catalog="main",
    schema="langgraph",
)
saver.setup()
print("✓ 테이블 생성 완료")

# COMMAND ----------

# MAGIC %md ## Step 2 — MLflow로 Agent 패키징
# MAGIC
# MAGIC `ChatDatabricks`는 Model Serving 환경에서 워크스페이스 인증을 자동으로 사용합니다.
# MAGIC OpenAI API 키나 별도 시크릿 설정이 필요 없습니다.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from langchain_core.tools import tool


@tool
def get_product_info(product_id: str) -> str:
    """상품 정보를 조회합니다."""
    catalog = {
        "P001": "MacBook Pro 14인치, 가격: 2,990,000원",
        "P002": "iPhone 15 Pro, 가격: 1,550,000원",
    }
    return catalog.get(product_id, f"상품 {product_id}를 찾을 수 없습니다.")


class LangGraphAgentModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel로 래핑된 LangGraph Agent.
    ChatDatabricks + DatabricksSQLCheckpointSaver 조합.
    """

    def load_context(self, context):
        """서빙 컨테이너 시작 시 한 번 실행됩니다."""
        from databricks_langchain import ChatDatabricks
        from langchain.agents import create_agent  # LangGraph v1 표준 API
        from checkpointers import DatabricksSQLCheckpointSaver

        # SQL Connector: DATABRICKS_HOST / DATABRICKS_TOKEN 자동 주입
        self.saver = DatabricksSQLCheckpointSaver(
            http_path=context.model_config["sql_warehouse_http_path"],
            catalog=context.model_config.get("catalog", "main"),
            schema=context.model_config.get("schema", "langgraph"),
        )

        # ChatDatabricks: 워크스페이스 인증 자동 적용, 외부 API 키 불필요
        llm = ChatDatabricks(
            endpoint=context.model_config.get(
                "llm_endpoint", "databricks-meta-llama-3-3-70b-instruct"
            ),
            temperature=0,
        )

        self.agent = create_agent(
            model=llm,
            tools=[get_product_info],
            checkpointer=self.saver,
            system_prompt="당신은 상품 정보를 안내하는 어시스턴트입니다.",
        )

    def predict(self, context, model_input, params=None):
        """
        요청 형식:
            {
                "thread_id": "user-42",
                "messages": [{"role": "user", "content": "P001 알려줘"}]
            }
        """
        import pandas as pd
        from langchain_core.messages import HumanMessage

        row = model_input.iloc[0].to_dict() if isinstance(model_input, pd.DataFrame) else model_input

        thread_id = row.get("thread_id", "default")
        messages = row.get("messages", [])

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=last_user_msg)]},
            config=config,
        )

        return {
            "thread_id": thread_id,
            "response": result["messages"][-1].content,
        }


# COMMAND ----------

# MAGIC %md ## Step 3 — MLflow에 모델 등록

# COMMAND ----------

mlflow.set_experiment("/Shared/langgraph-agent-experiment")

model_config = {
    "sql_warehouse_http_path": SQL_WAREHOUSE_HTTP_PATH,
    "catalog": "main",
    "schema": "langgraph",
    "llm_endpoint": "databricks-meta-llama-3-3-70b-instruct",
}

with mlflow.start_run(run_name="langgraph-agent-with-memory"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="langgraph_agent",
        python_model=LangGraphAgentModel(),
        model_config=model_config,
        pip_requirements=[
            "langchain>=0.3.0",
            "langgraph>=1.0.0",
            "databricks-langchain>=0.1.0",
            "databricks-sql-connector>=3.0.0",
        ],
        registered_model_name="langgraph_agent_with_memory",
    )
    print(f"✓ 모델 등록 완료: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md ## Step 4 — Model Serving 엔드포인트 생성
# MAGIC
# MAGIC `ChatDatabricks` 사용 시 `OPENAI_API_KEY` 환경변수가 필요 없습니다.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedModelInput, EndpointCoreConfigInput

w = WorkspaceClient()

endpoint = w.serving_endpoints.create_and_wait(
    name="langgraph-agent-memory",
    config=EndpointCoreConfigInput(
        served_models=[
            ServedModelInput(
                model_name="langgraph_agent_with_memory",
                model_version="1",
                scale_to_zero_enabled=True,
                workload_size="Small",
                # 외부 API 키 불필요 — ChatDatabricks가 워크스페이스 인증 자동 사용
            )
        ]
    ),
)
print(f"✓ 엔드포인트 생성 완료: {endpoint.state}")

# COMMAND ----------

# MAGIC %md ## Step 5 — 엔드포인트 호출 테스트

# COMMAND ----------

import requests

ENDPOINT_URL = f"{w.config.host}/serving-endpoints/langgraph-agent-memory/invocations"
HEADERS = {"Authorization": f"Bearer {w.config.token}", "Content-Type": "application/json"}


def ask(thread_id: str, message: str) -> str:
    payload = {
        "dataframe_records": [
            {"thread_id": thread_id, "messages": [{"role": "user", "content": message}]}
        ]
    }
    res = requests.post(ENDPOINT_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()["predictions"][0]["response"]


print(ask("user-001", "P001 상품 정보 알려줘"))
print(ask("user-001", "방금 알려준 상품 가격이 얼마야?"))  # Delta에서 이전 대화 복원
