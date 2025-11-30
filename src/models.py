from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel


class AggregatedBlock(BaseModel):
    id: int
    post_ids: List[int]
    insight_ids: List[int]
    title: str
    text: str


class Chunk(BaseModel):
    id: int
    post_id: int
    text: str


class Insight(BaseModel):
    id: int
    post_id: int
    text: str
    channel_category: List[Dict[str, str]]


class Post(BaseModel):
    id: int
    date: str
    channel_id: int
    forwards: int
    views: int
    text: str
    channel_category: List[Dict[str, str]]
    reactions: Optional[Dict[str, Any]]


class RawBlock(BaseModel):
    id: int
    post_id: int
    text: str


class PriceLabel(BaseModel):
    """Целевая метка направления изменения цены."""
    ticker: str
    timestamp: datetime
    base_price: float  # Close_t
    target_price: float  # Close_{t+H}
    direction: str  # "up" или "down"
    horizon_hours: int  # H в часах


class NewsFeature(BaseModel):
    """Признаки новости, извлеченные через LLM."""
    post_id: int
    ticker: Optional[str]  # Упомянутый тикер
    sentiment: Optional[str]  # Тональность (positive, negative, neutral)
    urgency: Optional[str]  # Срочность (high, medium, low)
    event_type: Optional[str]  # Тип события
    mentions: List[str]  # Упоминания тикеров/персон
    direction_prediction: Optional[str]  # Прогноз направления (up/down)
    raw_features: Dict[str, Any]  # Дополнительные признаки


class Prediction(BaseModel):
    """Прогноз направления изменения цены."""
    ticker: str
    timestamp: datetime
    news_window_start: datetime
    news_window_end: datetime
    predicted_direction: str  # "up" или "down"
    confidence: Optional[float]  # Уверенность (0-1)
    features: List[NewsFeature]
    method: str  # "zero-shot", "few-shot", "classifier"

