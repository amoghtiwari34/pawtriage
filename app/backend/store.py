"""SQLite models via sqlmodel."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

import os
_RT = Path("/tmp/pawtriage") if os.environ.get("VERCEL") else Path(__file__).resolve().parent
_RT.mkdir(parents=True, exist_ok=True)
DB = _RT / "pawtriage.db"
engine = create_engine(f"sqlite:///{DB}", connect_args={"check_same_thread": False})


def now() -> datetime:
    return datetime.now(timezone.utc)


class Pet(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    species: str
    breed: str = ""
    age_years: float = 0
    weight_kg: float = 0
    meds: str = "[]"          # JSON list
    conditions: str = "[]"    # JSON list

    def profile(self) -> dict:
        return {"name": self.name, "age_years": self.age_years, "breed": self.breed, "weight_kg": self.weight_kg,
                "meds": json.loads(self.meds), "conditions": json.loads(self.conditions)}


class Case(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pet_id: int = Field(foreign_key="pet.id")
    created_at: datetime = Field(default_factory=now)
    recheck_due: Optional[datetime] = None
    status: str = "open"
    turns: str = "[]"          # JSON list of {at, message, media, result, audit, model, cost_usd, latency_ms}
    checked: str = "[]"        # JSON list of plan items the owner ticked
    clock_offset_h: float = 0  # demo time travel


class Outcome(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: int = Field(foreign_key="case.id")
    at: datetime = Field(default_factory=now)
    message: str
    prev_urgency: str = ""
    new_urgency: str = ""


def init() -> None:
    SQLModel.metadata.create_all(engine)


def session() -> Session:
    return Session(engine)
