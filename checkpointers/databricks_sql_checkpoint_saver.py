"""
DatabricksSQLCheckpointSaver
============================
LangGraph BaseCheckpointSaver backed by Delta Lake,
using **Databricks SQL Connector** instead of SparkSession.

SparkSession이 없는 환경을 위한 구현체:
  - Databricks Model Serving 엔드포인트
  - 로컬 개발 환경 (databricks-connect 없이)
  - 일반 Python 서버/컨테이너

SQL Warehouse에 HTTP로 연결해 Delta 테이블을 읽고 씁니다.
Model Serving 환경에서는 DATABRICKS_HOST / DATABRICKS_TOKEN이
자동으로 주입되므로 별도 인증 설정이 필요 없습니다.

Install
-------
    pip install databricks-sql-connector langgraph langchain-openai

Usage (Model Serving endpoint)
------------------------------
    import mlflow
    from checkpointers import DatabricksSQLCheckpointSaver

    saver = DatabricksSQLCheckpointSaver(
        http_path="/sql/1.0/warehouses/<warehouse-id>",
        catalog="main",
        schema="langgraph",
    )
    # server_hostname / access_token 은 env var에서 자동 로드
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple

import databricks.sql as dbsql
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


def _strip_scheme(host: str) -> str:
    """'https://adb-xxx.net' → 'adb-xxx.net'"""
    return host.replace("https://", "").replace("http://", "").rstrip("/")


class DatabricksSQLCheckpointSaver(BaseCheckpointSaver):
    """
    LangGraph checkpoint saver that persists state in Delta Lake,
    accessed via Databricks SQL Connector (no SparkSession required).

    Parameters
    ----------
    http_path:
        SQL Warehouse HTTP path.
        예: ``"/sql/1.0/warehouses/abc123def456"``
        Databricks 워크스페이스 → SQL Warehouses → Connection details에서 확인.
    catalog:
        Unity Catalog 카탈로그 이름 (기본값: ``"main"``).
    schema:
        스키마 이름 (기본값: ``"langgraph"``).
    table_prefix:
        테이블 이름 접두사 (기본값: ``"langgraph"``).
    server_hostname:
        Databricks 워크스페이스 호스트명.
        미지정 시 ``DATABRICKS_HOST`` 환경변수 사용.
        Model Serving에서는 자동 주입됩니다.
    access_token:
        Personal Access Token 또는 서비스 주체 토큰.
        미지정 시 ``DATABRICKS_TOKEN`` 환경변수 사용.
        Model Serving에서는 자동 주입됩니다.
    """

    def __init__(
        self,
        http_path: str,
        catalog: str = "main",
        schema: str = "langgraph",
        table_prefix: str = "langgraph",
        server_hostname: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        super().__init__(serde=JsonPlusSerializer())

        self._http_path = http_path

        host = server_hostname or os.getenv("DATABRICKS_HOST")
        if not host:
            raise ValueError(
                "server_hostname이 지정되지 않았고 DATABRICKS_HOST 환경변수도 없습니다.\n"
                "로컬 테스트 시: os.environ['DATABRICKS_HOST'] = 'https://adb-xxx.azuredatabricks.net'\n"
                "Model Serving에서는 자동 주입됩니다."
            )
        self._server_hostname = _strip_scheme(host)

        self._access_token = access_token or os.getenv("DATABRICKS_TOKEN")

        self.catalog = catalog
        self.schema = schema
        self.table_prefix = table_prefix

        fq = f"`{catalog}`.`{schema}`"
        self._cp_table = f"{fq}.`{table_prefix}_checkpoints`"
        self._wr_table = f"{fq}.`{table_prefix}_checkpoint_writes`"

        # 스레드별 독립 커넥션 (Model Serving은 멀티스레드로 동작)
        self._local = threading.local()

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    def _get_connection(self):
        """현재 스레드의 SQL 커넥션을 반환. 없으면 새로 생성."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = dbsql.connect(
                server_hostname=self._server_hostname,
                http_path=self._http_path,
                access_token=self._access_token,
                session_configuration={"ansi_mode": "false"},
            )
            self._local.conn = conn
            logger.debug("SQL connection opened (thread=%s)", threading.get_ident())
        return conn

    @contextmanager
    def _cursor(self):
        cursor = self._get_connection().cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def close(self) -> None:
        """현재 스레드의 커넥션을 닫습니다. 명시적 정리가 필요할 때 사용."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def setup(self) -> None:
        """
        Delta 테이블을 생성합니다. 이미 존재하면 스킵합니다.
        매 실행마다 호출해도 안전합니다 (idempotent).
        """
        with self._cursor() as cur:
            cur.execute(
                f"CREATE SCHEMA IF NOT EXISTS `{self.catalog}`.`{self.schema}`"
            )
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._cp_table} (
                    thread_id            STRING    NOT NULL,
                    checkpoint_ns        STRING    NOT NULL,
                    checkpoint_id        STRING    NOT NULL,
                    parent_checkpoint_id STRING,
                    type                 STRING    NOT NULL,
                    checkpoint           STRING    NOT NULL,
                    metadata             STRING    NOT NULL,
                    created_at           TIMESTAMP NOT NULL
                )
                USING DELTA
                PARTITIONED BY (thread_id)
                TBLPROPERTIES (
                    'delta.autoOptimize.optimizeWrite' = 'true',
                    'delta.autoOptimize.autoCompact'   = 'true'
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._wr_table} (
                    thread_id     STRING  NOT NULL,
                    checkpoint_ns STRING  NOT NULL,
                    checkpoint_id STRING  NOT NULL,
                    task_id       STRING  NOT NULL,
                    idx           INT     NOT NULL,
                    channel       STRING  NOT NULL,
                    type          STRING  NOT NULL,
                    value         STRING
                )
                USING DELTA
                PARTITIONED BY (thread_id)
                TBLPROPERTIES (
                    'delta.autoOptimize.optimizeWrite' = 'true',
                    'delta.autoOptimize.autoCompact'   = 'true'
                )
            """)
        logger.info(
            "DatabricksSQLCheckpointSaver tables ready: %s, %s",
            self._cp_table,
            self._wr_table,
        )

    # -----------------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------------

    def _ser_checkpoint(self, cp: Checkpoint) -> Tuple[str, str]:
        type_, bytes_ = self.serde.dumps_typed(cp)
        return type_, bytes_.decode()

    def _deser_checkpoint(self, type_: str, data: str) -> Checkpoint:
        return self.serde.loads_typed((type_, data.encode()))

    def _ser_metadata(self, meta: CheckpointMetadata) -> str:
        _, bytes_ = self.serde.dumps_typed(meta)
        return bytes_.decode()

    def _deser_metadata(self, data: str) -> CheckpointMetadata:
        return self.serde.loads_typed(("json", data.encode()))

    def _deser_writes(self, rows) -> list[PendingWrite]:
        return [
            (
                row["task_id"],
                row["channel"],
                self.serde.loads_typed((row["type"], row["value"].encode())),
            )
            for row in rows
        ]

    # -----------------------------------------------------------------------
    # Config helpers
    # -----------------------------------------------------------------------

    def _current_config(self, row: dict) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["checkpoint_id"],
            }
        }

    def _parent_config(self, row: dict) -> Optional[RunnableConfig]:
        if not row.get("parent_checkpoint_id"):
            return None
        return {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["parent_checkpoint_id"],
            }
        }

    def _rows_to_dicts(self, cursor) -> list[dict]:
        """cursor.fetchall() 결과를 컬럼명 기준 dict 리스트로 변환."""
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # Pending writes fetch
    # -----------------------------------------------------------------------

    def _fetch_writes(
        self, cursor, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[PendingWrite]:
        cursor.execute(
            f"""
            SELECT task_id, channel, type, value
            FROM {self._wr_table}
            WHERE thread_id = %s
              AND checkpoint_ns = %s
              AND checkpoint_id = %s
            ORDER BY task_id, idx
            """,
            [thread_id, checkpoint_ns, checkpoint_id],
        )
        rows = self._rows_to_dicts(cursor)
        return self._deser_writes(rows)

    def _row_to_tuple(self, row: dict, cursor) -> CheckpointTuple:
        pending_writes = self._fetch_writes(
            cursor, row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]
        )
        return CheckpointTuple(
            config=self._current_config(row),
            checkpoint=self._deser_checkpoint(row["type"], row["checkpoint"]),
            metadata=self._deser_metadata(row["metadata"]),
            parent_config=self._parent_config(row),
            pending_writes=pending_writes,
        )

    # -----------------------------------------------------------------------
    # BaseCheckpointSaver — synchronous interface
    # -----------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """최신(또는 특정) 체크포인트를 조회합니다."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: Optional[str] = get_checkpoint_id(config)

        with self._cursor() as cur:
            if checkpoint_id:
                cur.execute(
                    f"""
                    SELECT * FROM {self._cp_table}
                    WHERE thread_id = %s
                      AND checkpoint_ns = %s
                      AND checkpoint_id = %s
                    LIMIT 1
                    """,
                    [thread_id, checkpoint_ns, checkpoint_id],
                )
            else:
                cur.execute(
                    f"""
                    SELECT * FROM {self._cp_table}
                    WHERE thread_id = %s
                      AND checkpoint_ns = %s
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    [thread_id, checkpoint_ns],
                )

            rows = self._rows_to_dicts(cur)
            if not rows:
                return None

            return self._row_to_tuple(rows[0], cur)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """체크포인트 히스토리를 최신순으로 순회합니다."""
        conditions: list[str] = []
        params: list[Any] = []

        if config is not None:
            conditions.append("thread_id = %s")
            params.append(config["configurable"]["thread_id"])
            conditions.append("checkpoint_ns = %s")
            params.append(config["configurable"].get("checkpoint_ns", ""))

        if before is not None:
            before_id = get_checkpoint_id(before)
            if before_id:
                conditions.append("checkpoint_id < %s")
                params.append(before_id)

        # metadata JSON 컬럼 필터 (예: filter={"source": "loop"})
        if filter:
            for key, value in filter.items():
                conditions.append(f"GET_JSON_OBJECT(metadata, '$.{key}') = %s")
                params.append(str(value))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""

        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {self._cp_table}
                {where}
                ORDER BY checkpoint_id DESC
                {limit_clause}
                """,
                params,
            )
            rows = self._rows_to_dicts(cur)

            for row in rows:
                yield self._row_to_tuple(row, cur)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """체크포인트를 Delta 테이블에 저장합니다."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = checkpoint["id"]
        parent_id: Optional[str] = config["configurable"].get("checkpoint_id")

        type_, cp_data = self._ser_checkpoint(checkpoint)
        meta_data = self._ser_metadata(metadata)
        now = datetime.now(tz=timezone.utc).isoformat()

        with self._cursor() as cur:
            # MERGE INTO — 같은 checkpoint_id가 두 번 쓰여도 중복 없음
            cur.execute(
                f"""
                MERGE INTO {self._cp_table} AS target
                USING (
                    SELECT
                        %s AS thread_id,
                        %s AS checkpoint_ns,
                        %s AS checkpoint_id,
                        %s AS parent_checkpoint_id,
                        %s AS type,
                        %s AS checkpoint,
                        %s AS metadata,
                        CAST(%s AS TIMESTAMP) AS created_at
                ) AS source
                ON  target.thread_id     = source.thread_id
                AND target.checkpoint_ns = source.checkpoint_ns
                AND target.checkpoint_id = source.checkpoint_id
                WHEN NOT MATCHED THEN INSERT *
                WHEN MATCHED THEN UPDATE SET
                    type       = source.type,
                    checkpoint = source.checkpoint,
                    metadata   = source.metadata,
                    created_at = source.created_at
                """,
                [
                    thread_id, checkpoint_ns, checkpoint_id,
                    parent_id,
                    type_, cp_data, meta_data, now,
                ],
            )

        logger.debug("put checkpoint %s / thread=%s", checkpoint_id, thread_id)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """노드의 중간 쓰기(Pending Writes)를 저장합니다."""
        if not writes:
            return

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]

        with self._cursor() as cur:
            for idx, (channel, value) in enumerate(writes):
                type_, bytes_ = self.serde.dumps_typed(value)
                value_str = bytes_.decode()

                cur.execute(
                    f"""
                    MERGE INTO {self._wr_table} AS target
                    USING (
                        SELECT
                            %s AS thread_id,
                            %s AS checkpoint_ns,
                            %s AS checkpoint_id,
                            %s AS task_id,
                            %s AS idx,
                            %s AS channel,
                            %s AS type,
                            %s AS value
                    ) AS source
                    ON  target.thread_id     = source.thread_id
                    AND target.checkpoint_ns = source.checkpoint_ns
                    AND target.checkpoint_id = source.checkpoint_id
                    AND target.task_id       = source.task_id
                    AND target.idx           = source.idx
                    WHEN NOT MATCHED THEN INSERT *
                    WHEN MATCHED THEN UPDATE SET
                        channel = source.channel,
                        type    = source.type,
                        value   = source.value
                    """,
                    [
                        thread_id, checkpoint_ns, checkpoint_id,
                        task_id, idx, channel, type_, value_str,
                    ],
                )

        logger.debug(
            "put_writes %d writes / checkpoint=%s / thread=%s",
            len(writes), checkpoint_id, thread_id,
        )

    # -----------------------------------------------------------------------
    # Async interface — asyncio.to_thread으로 동기 메서드를 비동기 실행
    # SQL Connector가 동기 전용이므로 스레드 풀에서 실행
    # -----------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        """특정 스레드의 모든 체크포인트와 쓰기 데이터를 삭제합니다."""
        with self._cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._cp_table} WHERE thread_id = %s",
                [thread_id],
            )
            cur.execute(
                f"DELETE FROM {self._wr_table} WHERE thread_id = %s",
                [thread_id],
            )
        logger.info("Deleted all checkpoints for thread_id=%s", thread_id)
