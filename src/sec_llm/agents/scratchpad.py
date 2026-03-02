"""Pydantic state models for the multi-agent scratchpad."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class AgentRole(str, Enum):
    RESEARCHER = "RESEARCHER"
    QUANT = "QUANT"
    VALIDATOR = "VALIDATOR"
    BOSS = "BOSS"


class ScratchpadTask(BaseModel):
    task_id: str
    assigned_to: AgentRole
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result_key: str | None = None
    depends_on: list[str] = []
    error_message: str | None = None


class RevisionEntry(BaseModel):
    timestamp: str
    reason: str
    tasks_added: list[str] = []
    tasks_modified: list[str] = []


class Scratchpad(BaseModel):
    goal: str
    tasks: list[ScratchpadTask] = []
    knowledge_base: dict[str, Any] = {}
    revision_history: list[RevisionEntry] = []


class CitationEntry(BaseModel):
    ticker: str
    filing_type: str
    fiscal_period: str
    filing_date: str


class BossResponse(BaseModel):
    final_answer: str
    scratchpad: Scratchpad
    citations: list[CitationEntry] = []


class SingleAgentResponse(BaseModel):
    final_answer: str
    citations: list[CitationEntry] = []
