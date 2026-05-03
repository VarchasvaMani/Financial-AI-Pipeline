"""
Core data models and database setup for the financial AI pipeline.
Uses SQLite for local development — swap for PostgreSQL/TimescaleDB in production.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import DeclarativeBase, Session
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/pipeline.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class Base(DeclarativeBase):
    pass


# ─────────────────────────── ORM Tables ────────────────────────────

class OHLCVRecord(Base):
    __tablename__ = "ohlcv"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    symbol       = Column(String(10), nullable=False, index=True)
    timestamp    = Column(DateTime, nullable=False, index=True)
    open         = Column(Float)
    high         = Column(Float)
    low          = Column(Float)
    close        = Column(Float)
    volume       = Column(Float)
    source       = Column(String(50), default="yfinance")


class FeatureRecord(Base):
    __tablename__ = "features"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    symbol          = Column(String(10), nullable=False, index=True)
    timestamp       = Column(DateTime, nullable=False, index=True)
    rsi             = Column(Float)
    macd            = Column(Float)
    macd_signal     = Column(Float)
    bb_upper        = Column(Float)
    bb_lower        = Column(Float)
    bb_mid          = Column(Float)
    atr             = Column(Float)
    volume_zscore   = Column(Float)
    realized_vol    = Column(Float)
    momentum_1d     = Column(Float)
    momentum_5d     = Column(Float)
    momentum_20d    = Column(Float)
    vwap            = Column(Float)


class NewsRecord(Base):
    __tablename__ = "news"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    symbol       = Column(String(10), nullable=False, index=True)
    timestamp    = Column(DateTime, nullable=False, index=True)
    headline     = Column(Text)
    source       = Column(String(100))
    url          = Column(Text)
    sentiment    = Column(String(20))   # bullish / bearish / neutral
    magnitude    = Column(Integer)      # 1-5
    theme        = Column(Text)
    confidence   = Column(Float)
    processed    = Column(Boolean, default=False)


class SignalRecord(Base):
    __tablename__ = "signals"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    symbol          = Column(String(10), nullable=False, index=True)
    timestamp       = Column(DateTime, nullable=False, index=True)
    direction       = Column(String(10))   # long / short / neutral
    confidence      = Column(Float)
    horizon         = Column(String(10))   # 1h / 4h / 1d
    top_features    = Column(Text)         # JSON string of top SHAP features
    narrative       = Column(Text)         # Claude-generated explanation
    sentiment_score = Column(Float)        # aggregated news sentiment


class AuditRecord(Base):
    __tablename__ = "audit"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    event_type  = Column(String(50))
    symbol      = Column(String(10))
    details     = Column(Text)
    model_ver   = Column(String(20))


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    print(f"[DB] Initialised at {DB_PATH}")


def get_session() -> Session:
    return Session(engine)


# ─────────────────────────── Pydantic schemas ───────────────────────

class MarketEvent(BaseModel):
    symbol:    str
    timestamp: datetime
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    source:    str = "yfinance"

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        return v.upper()


class SentimentResult(BaseModel):
    sentiment:  str    # bullish / bearish / neutral
    magnitude:  int    # 1–5
    theme:      str
    confidence: float


class SignalResult(BaseModel):
    symbol:          str
    direction:       str
    confidence:      float
    horizon:         str
    top_features:    list[tuple[str, float]]
    narrative:       str
    sentiment_score: float


if __name__ == "__main__":
    init_db()
    print("Tables created successfully.")
