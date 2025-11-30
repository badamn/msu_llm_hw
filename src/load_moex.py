"""Модуль для загрузки котировок MOEX."""
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pytz


class MOEXClient:
    """Клиент для загрузки исторических котировок MOEX."""
    
    BASE_URL = "https://iss.moex.com/iss"
    MSK_TZ = pytz.timezone("Europe/Moscow")
    
    # Время торговых сессий MOEX (MSK)
    TRADING_SESSION_START = 9  # 9:00 MSK
    TRADING_SESSION_END = 18  # 18:45 MSK (обычно до 18:45)
    
    def __init__(self):
        self.session = requests.Session()
    
    def _get_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Получает список торговых дней между датами."""
        # Упрощенная версия - считаем все будние дни торговыми
        # В реальности нужно учитывать календарь торгов MOEX
        trading_days = []
        current = start_date.date()
        end = end_date.date()
        
        while current <= end:
            # Пропускаем выходные (суббота=5, воскресенье=6)
            if current.weekday() < 5:
                trading_days.append(datetime.combine(current, datetime.min.time()).replace(tzinfo=self.MSK_TZ))
            current += timedelta(days=1)
        
        return trading_days
    
    def _get_next_trading_day_close(self, dt: datetime) -> datetime:
        """Получает время закрытия следующего торгового дня после dt."""
        dt = dt.astimezone(self.MSK_TZ)
        current_date = dt.date()
        
        # Если запрос после закрытия торгов, следующий торговый день - завтра
        if dt.hour >= self.TRADING_SESSION_END:
            next_date = current_date + timedelta(days=1)
        else:
            next_date = current_date
        
        # Пропускаем выходные
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        
        # Время закрытия торгового дня (18:45 MSK)
        from datetime import time as dt_time
        close_time = datetime.combine(next_date, dt_time(hour=18, minute=45))
        return self.MSK_TZ.localize(close_time)
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "24"  # "1", "60", "24" (минуты, часы, дни)
    ) -> pd.DataFrame:
        """
        Загружает исторические котировки OHLCV для тикера.
        
        Args:
            ticker: Тикер акции (например, "GAZP", "SBER", "LKOH")
            start_date: Начальная дата
            end_date: Конечная дата
            interval: Интервал данных ("1", "60", "24")
        
        Returns:
            DataFrame с колонками: TRADEDATE, OPEN, HIGH, LOW, CLOSE, VOLUME
        """
        start_date = start_date.astimezone(self.MSK_TZ)
        end_date = end_date.astimezone(self.MSK_TZ)
        
        url = f"{self.BASE_URL}/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "till": end_date.strftime("%Y-%m-%d"),
            "interval": interval,
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if "history" not in data or "data" not in data["history"]:
            return pd.DataFrame()
        
        columns = data["history"]["columns"]
        rows = data["history"]["data"]
        
        df = pd.DataFrame(rows, columns=columns)
        
        if df.empty:
            return df
        
        # Конвертируем дату
        if "TRADEDATE" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        
        # Переименовываем колонки для удобства
        column_mapping = {
            "TRADEDATE": "date",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
        }
        
        df = df.rename(columns=column_mapping)
        
        # Фильтруем только нужные колонки
        available_cols = [col for col in column_mapping.values() if col in df.columns]
        df = df[available_cols]
        
        # Устанавливаем дату как индекс
        if "date" in df.columns:
            df = df.set_index("date")
            df.index = pd.to_datetime(df.index).tz_localize(self.MSK_TZ)
        
        return df
    
    def get_close_price(self, ticker: str, dt: datetime) -> Optional[float]:
        """
        Получает цену закрытия для конкретной даты/времени.
        
        Args:
            ticker: Тикер акции
            dt: Дата и время (MSK)
        
        Returns:
            Цена закрытия или None если данных нет
        """
        dt = dt.astimezone(self.MSK_TZ)
        date = dt.date()
        
        # Загружаем данные за день
        start = datetime.combine(date, datetime.min.time()).replace(tzinfo=self.MSK_TZ)
        end = datetime.combine(date, datetime.max.time()).replace(tzinfo=self.MSK_TZ)
        
        df = self.get_ohlcv(ticker, start, end, interval="24")
        
        if df.empty or "close" not in df.columns:
            return None
        
        # Берем последнюю доступную цену закрытия
        return float(df["close"].iloc[-1])
    
    def get_price_change_direction(
        self, ticker: str, base_time: datetime, target_time: datetime
    ) -> Optional[str]:
        """
        Определяет направление изменения цены между двумя точками времени.
        
        Args:
            ticker: Тикер акции
            base_time: Базовое время (Close_t)
            target_time: Целевое время (Close_{t+H})
        
        Returns:
            "up" если цена выросла, "down" если упала, None если данных нет
        """
        base_price = self.get_close_price(ticker, base_time)
        target_price = self.get_close_price(ticker, target_time)
        
        if base_price is None or target_price is None:
            return None
        
        if target_price > base_price:
            return "up"
        elif target_price < base_price:
            return "down"
        else:
            return "up"  # При равенстве считаем "up" (можно изменить логику)
    
    def is_trading_session(self, dt: datetime) -> bool:
        """Проверяет, является ли время торговой сессией."""
        dt = dt.astimezone(self.MSK_TZ)
        
        # Проверяем день недели (пн-пт)
        if dt.weekday() >= 5:
            return False
        
        # Проверяем время
        hour = dt.hour
        return self.TRADING_SESSION_START <= hour < self.TRADING_SESSION_END

