# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving 엔드포인트에 LangGraph Agent 배포
# MAGIC
# MAGIC `DatabricksSQLCheckpointSaver`를 사용해 Model Serving 환경에서도
# MAGIC Delta Lake 기반 메모리를 유지합니다.
# MAGIC
# MAGIC ## 흐름
# MAGIC ```
# MAGIC [클라이언트]
# MAGIC     ↓  REST 요청 (thread_id 포함)
# MAGIC [Model Serving Endpoint]  ← SparkSession 없음
# MAGIC     ↓  DatabricksSQLCheckpointSaver
# MAGIC     ↓  Databricks SQL Connector (HTTP)
# MAGIC [SQL Warehouse]
# MAGIC     ↓
# MAGIC [Delta Lake: langgraph_checkpoints 테이블]
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install langgraph langchain-openai databricks-sql-connector mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Step 1 — 테이블 미리 생성 (노트북에서 한 번만)
# MAGIC
# MAGIC Model Serving 환경에는 `setup()` 권한이 없을 수 있으므로,
# MAGIC 배포 전 노트북에서 테이블을 만들어둡니다.

# COMMAND ----------

import os
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

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
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
    Model Serving에서 인스턴스화됩니다.
    """

    def load_context(self, context):
        """서빙 컨테이너 시작 시 한 번 실행됩니다."""
        import os
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
        from checkpointers import DatabricksSQLCheckpointSaver

        # Model Serving이 자동으로 주입하는 환경변수
        # DATABRICKS_HOST, DATABRICKS_TOKEN

        self.saver = DatabricksSQLCheckpointSaver(
            http_path=context.model_config["sql_warehouse_http_path"],
            catalog=context.model_config.get("catalog", "main"),
            schema=context.model_config.get("schema", "langgraph"),
        )

        llm = ChatOpenAI(
            model=context.model_config.get("llm_model", "gpt-4o-mini"),
            temperature=0,
        )
        self.agent = create_react_agent(llm, [get_product_info], checkpointer=self.saver)

    def predict(self, context, model_input, params=None):
        """
        요청 형식:
            {
                "messages": [{"role": "user", "content": "안녕하세요"}],
                "thread_id": "user-42"
            }
        """
        import pandas as pd
        from langchain_core.messages import HumanMessage, AIMessage

        if isinstance(model_input, pd.DataFrame):
            row = model_input.iloc[0].to_dict()
        else:
            row = model_input

        thread_id = row.get("thread_id", "default")
        messages = row.get("messages", [])

        # 마지막 user 메시지만 LangGraph에 전달 (히스토리는 체크포인트에서 복원)
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
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
    "llm_model": "gpt-4o-mini",
}

with mlflow.start_run(run_name="langgraph-agent-with-memory"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="langgraph_agent",
        python_model=LangGraphAgentModel(),
        model_config=model_config,
        pip_requirements=[
            "langgraph>=0.2.0",
            "langchain-openai>=0.1.0",
            "databricks-sql-connector>=3.0.0",
        ],
        registered_model_name="langgraph_agent_with_memory",
    )
    print(f"✓ 모델 등록 완료: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md ## Step 4 — Model Serving 엔드포인트 생성
# MAGIC
# MAGIC 워크스페이스 UI에서:
# MAGIC 1. **Serving** → **Create serving endpoint**
# MAGIC 2. Entity: `langgraph_agent_with_memory` 선택
# MAGIC 3. Environment variables 추가:
# MAGIC    - `OPENAI_API_KEY` = `{{secrets/my-scope/openai-api-key}}`
# MAGIC 4. **Create**

# COMMAND ----------

# Databricks SDK로 엔드포인트 생성 자동화
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedModelInput,
    EndpointCoreConfigInput,
    EnvVariable,
)

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
                environment_vars=[
                    EnvVariable(
                        key="OPENAI_API_KEY",
                        value="{{secrets/my-scope/openai-api-key}}",
                    )
                ],
            )
        ]
    ),
)
print(f"✓ 엔드포인트 생성 완료: {endpoint.state}")

# COMMAND ----------

# MAGIC %md ## Step 5 — 엔드포인트 호출 테스트
# MAGIC
# MAGIC `thread_id`가 같으면 Delta 테이블에서 이전 대화가 복원됩니다.

# COMMAND ----------

import requests

ENDPOINT_URL = f"{w.config.host}/serving-endpoints/langgraph-agent-memory/invocations"
HEADERS = {"Authorization": f"Bearer {w.config.token}", "Content-Type": "application/json"}

def ask(thread_id: str, message: str) -> str:
    payload = {
        "dataframe_records": [
            {
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": message}],
            }
        ]
    }
    res = requests.post(ENDPOINT_URL, headers=HEADERS, json=payload)
    res.raise_for_status()
    return res.json()["predictions"][0]["response"]

# 대화 1 — 첫 번째 메시지
print(ask("user-001", "P001 상품 정보 알려줘"))
# 대화 2 — 같은 thread_id로 이어지는 대화 (Delta에서 복원)
print(ask("user-001", "방금 알려준 상품 가격이 얼마야?"))
