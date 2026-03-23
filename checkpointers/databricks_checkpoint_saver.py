"""
DatabricksCheckpointSaver
=========================
LangGraph BaseCheckpointSaver backed by Delta Lake on Databricks.

Stores agent checkpoints (conversation state) in two Delta tables so that
memory accumulates across sessions and is queryable with SQL.

Tables
------
{catalog}.{schema}.{prefix}_checkpoints       — one row per checkpoint
{catalog}.{schema}.{prefix}_checkpoint_writes — intermediate node writes

Quick start (Databricks notebook)
----------------------------------
    from checkpointers import DatabricksCheckpointSaver

    saver = DatabricksCheckpointSaver(spark, catalog="main", schema="langgraph")
    saver.setup()   # idempotent — safe to call every run

    graph = create_react_agent(model, tools, checkpointer=saver)
    config = {"configurable": {"thread_id": "user-42"}}
    result = graph.invoke({"messages": [HumanMessage("hello")]}, config)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple

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
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_CHECKPOINTS_STRUCT = StructType(
    [
        StructField("thread_id", StringType(), nullable=False),
        StructField("checkpoint_ns", StringType(), nullable=False),
        StructField("checkpoint_id", StringType(), nullable=False),
        StructField("parent_checkpoint_id", StringType(), nullable=True),
        StructField("type", StringType(), nullable=False),
        StructField("checkpoint", StringType(), nullable=False),
        StructField("metadata", StringType(), nullable=False),
        StructField("created_at", TimestampType(), nullable=False),
    ]
)

_WRITES_STRUCT = StructType(
    [
        StructField("thread_id", StringType(), nullable=False),
        StructField("checkpoint_ns", StringType(), nullable=False),
        StructField("checkpoint_id", StringType(), nullable=False),
        StructField("task_id", StringType(), nullable=False),
        StructField("idx", IntegerType(), nullable=False),
        StructField("channel", StringType(), nullable=False),
        StructField("type", StringType(), nullable=False),
        StructField("value", StringType(), nullable=True),
    ]
)


class DatabricksCheckpointSaver(BaseCheckpointSaver):
    """
    LangGraph checkpoint saver that persists state in Delta Lake tables.

    Parameters
    ----------
    spark:
        Active SparkSession (use ``SparkSession.builder.getOrCreate()``
        inside Databricks notebooks, or ``DatabricksSession`` via
        ``databricks-connect`` from a local machine).
    catalog:
        Unity Catalog catalog name (e.g. ``"main"``).
    schema:
        Schema / database name (e.g. ``"langgraph"``).
    table_prefix:
        Prefix for both tables (default ``"langgraph"``).
    """

    def __init__(
        self,
        spark: SparkSession,
        catalog: str = "main",
        schema: str = "langgraph",
        table_prefix: str = "langgraph",
    ) -> None:
        super().__init__(serde=JsonPlusSerializer())
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.table_prefix = table_prefix

        # Fully-qualified names for SQL statements
        fq = f"`{catalog}`.`{schema}`"
        self._cp_table = f"{fq}.`{table_prefix}_checkpoints`"
        self._wr_table = f"{fq}.`{table_prefix}_checkpoint_writes`"

        # Plain names for DataFrame API (.table() doesn't need backticks)
        self._cp_table_plain = f"{catalog}.{schema}.{table_prefix}_checkpoints"
        self._wr_table_plain = f"{catalog}.{schema}.{table_prefix}_checkpoint_writes"

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def setup(self) -> None:
        """
        Create the Delta Lake schema and tables if they do not exist.
        Idempotent — safe to call at the start of every notebook run.
        """
        self.spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS `{self.catalog}`.`{self.schema}`"
        )

        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self._cp_table} (
                thread_id            STRING    NOT NULL COMMENT 'LangGraph thread identifier',
                checkpoint_ns        STRING    NOT NULL COMMENT 'Namespace (empty for root, node:uuid for subgraphs)',
                checkpoint_id        STRING    NOT NULL COMMENT 'Unique monotonically-increasing checkpoint ID',
                parent_checkpoint_id STRING             COMMENT 'Parent checkpoint ID (NULL for first checkpoint)',
                type                 STRING    NOT NULL COMMENT 'Serializer type tag',
                checkpoint           STRING    NOT NULL COMMENT 'Serialized checkpoint JSON',
                metadata             STRING    NOT NULL COMMENT 'Serialized CheckpointMetadata JSON',
                created_at           TIMESTAMP NOT NULL COMMENT 'Wall-clock time of this checkpoint'
            )
            USING DELTA
            PARTITIONED BY (thread_id)
            TBLPROPERTIES (
                'delta.autoOptimize.optimizeWrite' = 'true',
                'delta.autoOptimize.autoCompact'   = 'true'
            )
        """)

        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self._wr_table} (
                thread_id     STRING  NOT NULL COMMENT 'LangGraph thread identifier',
                checkpoint_ns STRING  NOT NULL COMMENT 'Checkpoint namespace',
                checkpoint_id STRING  NOT NULL COMMENT 'Checkpoint this write belongs to',
                task_id       STRING  NOT NULL COMMENT 'Graph node task ID',
                idx           INT     NOT NULL COMMENT 'Write index within the task',
                channel       STRING  NOT NULL COMMENT 'Channel name written to',
                type          STRING  NOT NULL COMMENT 'Serializer type tag',
                value         STRING           COMMENT 'Serialized channel value'
            )
            USING DELTA
            PARTITIONED BY (thread_id)
            TBLPROPERTIES (
                'delta.autoOptimize.optimizeWrite' = 'true',
                'delta.autoOptimize.autoCompact'   = 'true'
            )
        """)

        logger.info(
            "DatabricksCheckpointSaver tables ready: %s, %s",
            self._cp_table,
            self._wr_table,
        )

    # -----------------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------------

    def _ser_checkpoint(self, cp: Checkpoint) -> Tuple[str, str]:
        type_, bytes_ = self.serde.dumps_typed(cp)
        return type_, bytes_.decode()

    def _deser_checkpoint(self, row: Row) -> Checkpoint:
        return self.serde.loads_typed((row["type"], row["checkpoint"].encode()))

    def _ser_metadata(self, meta: CheckpointMetadata) -> str:
        _, bytes_ = self.serde.dumps_typed(meta)
        return bytes_.decode()

    def _deser_metadata(self, row: Row) -> CheckpointMetadata:
        # metadata is always JSON-serialized
        return self.serde.loads_typed(("json", row["metadata"].encode()))

    def _deser_writes(self, rows: list[Row]) -> list[PendingWrite]:
        return [
            (
                r["task_id"],
                r["channel"],
                self.serde.loads_typed((r["type"], r["value"].encode())),
            )
            for r in rows
        ]

    # -----------------------------------------------------------------------
    # Config helpers
    # -----------------------------------------------------------------------

    def _current_config(self, row: Row) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["checkpoint_id"],
            }
        }

    def _parent_config(self, row: Row) -> Optional[RunnableConfig]:
        if not row["parent_checkpoint_id"]:
            return None
        return {
            "configurable": {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["parent_checkpoint_id"],
            }
        }

    # -----------------------------------------------------------------------
    # Internal fetch helpers (use DataFrame API — no SQL injection risk)
    # -----------------------------------------------------------------------

    def _fetch_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[PendingWrite]:
        rows = (
            self.spark.table(self._wr_table_plain)
            .filter(F.col("thread_id") == thread_id)
            .filter(F.col("checkpoint_ns") == checkpoint_ns)
            .filter(F.col("checkpoint_id") == checkpoint_id)
            .orderBy("task_id", "idx")
            .collect()
        )
        return self._deser_writes(rows)

    def _row_to_tuple(self, row: Row) -> CheckpointTuple:
        pending_writes = self._fetch_writes(
            row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]
        )
        return CheckpointTuple(
            config=self._current_config(row),
            checkpoint=self._deser_checkpoint(row),
            metadata=self._deser_metadata(row),
            parent_config=self._parent_config(row),
            pending_writes=pending_writes,
        )

    # -----------------------------------------------------------------------
    # BaseCheckpointSaver — synchronous interface
    # -----------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Return the latest (or specific) checkpoint for a thread."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: Optional[str] = get_checkpoint_id(config)

        df = (
            self.spark.table(self._cp_table_plain)
            .filter(F.col("thread_id") == thread_id)
            .filter(F.col("checkpoint_ns") == checkpoint_ns)
        )

        if checkpoint_id:
            df = df.filter(F.col("checkpoint_id") == checkpoint_id)
        else:
            df = df.orderBy(F.col("checkpoint_id").desc())

        rows = df.limit(1).collect()
        if not rows:
            return None

        return self._row_to_tuple(rows[0])

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Iterate checkpoints newest-first, with optional filters."""
        df = self.spark.table(self._cp_table_plain)

        if config is not None:
            thread_id: str = config["configurable"]["thread_id"]
            checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
            df = df.filter(F.col("thread_id") == thread_id).filter(
                F.col("checkpoint_ns") == checkpoint_ns
            )

        if before is not None:
            before_id = get_checkpoint_id(before)
            if before_id:
                df = df.filter(F.col("checkpoint_id") < before_id)

        # Metadata filters: use get_json_object for JSON column lookups
        if filter:
            for key, value in filter.items():
                df = df.filter(
                    F.get_json_object(F.col("metadata"), f"$.{key}") == str(value)
                )

        df = df.orderBy(F.col("checkpoint_id").desc())

        if limit is not None:
            df = df.limit(limit)

        for row in df.collect():
            yield self._row_to_tuple(row)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a new checkpoint to Delta Lake."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = checkpoint["id"]
        parent_id: Optional[str] = config["configurable"].get("checkpoint_id")

        type_, cp_data = self._ser_checkpoint(checkpoint)
        meta_data = self._ser_metadata(metadata)
        now = datetime.now(tz=timezone.utc)

        row = Row(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            parent_checkpoint_id=parent_id,
            type=type_,
            checkpoint=cp_data,
            metadata=meta_data,
            created_at=now,
        )

        # MERGE ensures idempotency — same checkpoint_id written twice is safe
        (
            self.spark.createDataFrame([row], schema=_CHECKPOINTS_STRUCT)
            .createOrReplaceTempView("_cp_upsert")
        )
        self.spark.sql(f"""
            MERGE INTO {self._cp_table} AS target
            USING _cp_upsert AS source
              ON  target.thread_id     = source.thread_id
              AND target.checkpoint_ns = source.checkpoint_ns
              AND target.checkpoint_id = source.checkpoint_id
            WHEN NOT MATCHED THEN INSERT *
            WHEN MATCHED THEN UPDATE SET
                type       = source.type,
                checkpoint = source.checkpoint,
                metadata   = source.metadata,
                created_at = source.created_at
        """)

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
        """Persist intermediate node writes for a checkpoint."""
        if not writes:
            return

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]

        rows = []
        for idx, (channel, value) in enumerate(writes):
            type_, bytes_ = self.serde.dumps_typed(value)
            rows.append(
                Row(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx=idx,
                    channel=channel,
                    type=type_,
                    value=bytes_.decode(),
                )
            )

        (
            self.spark.createDataFrame(rows, schema=_WRITES_STRUCT)
            .createOrReplaceTempView("_wr_upsert")
        )
        self.spark.sql(f"""
            MERGE INTO {self._wr_table} AS target
            USING _wr_upsert AS source
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
        """)

        logger.debug(
            "put_writes %d writes / checkpoint=%s / thread=%s",
            len(rows),
            checkpoint_id,
            thread_id,
        )

    # -----------------------------------------------------------------------
    # Async interface (delegates to sync — suitable for most Databricks use)
    # -----------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        return self.put_writes(config, writes, task_id)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        """Remove all checkpoints and writes for a thread."""
        self.spark.sql(f"""
            DELETE FROM {self._cp_table}
            WHERE thread_id = '{thread_id}'
        """)
        self.spark.sql(f"""
            DELETE FROM {self._wr_table}
            WHERE thread_id = '{thread_id}'
        """)
        logger.info("Deleted all checkpoints for thread_id=%s", thread_id)

    def get_thread_history(self, thread_id: str, limit: int = 20):
        """Return a Spark DataFrame of checkpoint history for a thread (for inspection)."""
        return (
            self.spark.table(self._cp_table_plain)
            .filter(F.col("thread_id") == thread_id)
            .orderBy(F.col("checkpoint_id").desc())
            .limit(limit)
            .select(
                "checkpoint_id",
                "parent_checkpoint_id",
                "created_at",
                F.get_json_object(F.col("metadata"), "$.source").alias("source"),
                F.get_json_object(F.col("metadata"), "$.step").alias("step"),
            )
        )
