"""Базовые утилиты для обработки текста и нормализации."""
import re
from typing import List, Set, Dict
from datetime import datetime
import pytz

from src.models import Post


class TextProcessor:
    """Обработчик текста для нормализации новостей."""
    
    # Синонимы и варианты названий тикеров
    TICKER_SYNONYMS = {
        "GAZP": ["газпром", "gazprom", "газпрома", "газпрому", "газпромом"],
        "SBER": ["сбер", "sberbank", "сбербанк", "сбербанка", "сбербанку", "сбербанком"],
        "LKOH": ["лукойл", "lukoil", "лукойла", "лукойлу", "лукойлом"],
    }
    
    # Регулярные выражения для удаления
    URL_PATTERN = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    
    def __init__(self):
        self.msk_tz = pytz.timezone("Europe/Moscow")
    
    def normalize_text(self, text: str) -> str:
        """
        Нормализует текст: удаляет ссылки, эмодзи, нормализует пробелы.
        
        Args:
            text: Исходный текст
        
        Returns:
            Нормализованный текст
        """
        # Удаляем ссылки
        text = self.URL_PATTERN.sub("", text)
        
        # Удаляем эмодзи
        text = self.EMOJI_PATTERN.sub("", text)
        
        # Удаляем хэштеги (опционально, можно оставить)
        # text = re.sub(r"#\w+", "", text)
        
        # Нормализуем пробелы
        text = re.sub(r"\s+", " ", text)
        
        # Удаляем лишние знаки препинания
        text = text.strip()
        
        return text
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Извлекает упоминания тикеров из текста.
        
        Args:
            text: Текст новости
        
        Returns:
            Список найденных тикеров
        """
        text_lower = text.lower()
        found_tickers = []
        
        # Проверяем прямые упоминания тикеров
        for ticker, synonyms in self.TICKER_SYNONYMS.items():
            if ticker.lower() in text_lower:
                found_tickers.append(ticker)
            else:
                # Проверяем синонимы
                for synonym in synonyms:
                    if synonym.lower() in text_lower:
                        found_tickers.append(ticker)
                        break
        
        return list(set(found_tickers))  # Убираем дубликаты
    
    def deduplicate_posts(self, posts: List[Post], similarity_threshold: float = 0.9) -> List[Post]:
        """
        Дедупликация постов по тексту и времени.
        
        Args:
            posts: Список постов
            similarity_threshold: Порог схожести (0-1)
        
        Returns:
            Список уникальных постов
        """
        if not posts:
            return []
        
        # Простая дедупликация по нормализованному тексту
        seen_texts: Set[str] = set()
        unique_posts = []
        
        for post in posts:
            normalized = self.normalize_text(post.text)
            normalized_lower = normalized.lower()
            
            # Проверяем, не видели ли мы похожий текст
            is_duplicate = False
            for seen_text in seen_texts:
                # Простая проверка на полное совпадение
                if normalized_lower == seen_text.lower():
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(normalized_lower)
                unique_posts.append(post)
        
        return unique_posts
    
    def filter_by_time_window(
        self, posts: List[Post], window_start: datetime, window_end: datetime, verbose: bool = False
    ) -> List[Post]:
        """
        Фильтрует посты по временному окну W.
        
        Args:
            posts: Список постов
            window_start: Начало окна
            window_end: Конец окна
            verbose: Выводить отладочную информацию
        
        Returns:
            Отфильтрованные посты
        """
        filtered = []
        errors = 0
        
        for post in posts:
            try:
                # Парсим дату поста - пробуем разные форматы
                post_date_str = post.date
                
                # Убираем Z и добавляем таймзону если нужно
                if "Z" in post_date_str:
                    post_date_str = post_date_str.replace("Z", "+00:00")
                elif "+" not in post_date_str and "Z" not in post_date_str:
                    # Если нет таймзоны, предполагаем UTC
                    post_date_str = post_date_str + "+00:00"
                
                post_dt = datetime.fromisoformat(post_date_str)
                if post_dt.tzinfo is None:
                    post_dt = pytz.utc.localize(post_dt)
                post_dt = post_dt.astimezone(self.msk_tz)
                
                # Проверяем попадание в окно
                if window_start <= post_dt <= window_end:
                    filtered.append(post)
                elif verbose and len(filtered) < 3:
                    # Показываем примеры постов вне окна
                    pass
            except Exception as e:
                errors += 1
                if verbose and errors <= 3:
                    print(f"      Ошибка парсинга даты поста {post.id}: {post.date} - {e}")
                continue
        
        if verbose:
            print(f"      Отфильтровано: {len(filtered)} из {len(posts)} постов попадают в окно")
            if errors > 0:
                print(f"      Ошибок парсинга дат: {errors}")
        
        return filtered
    
    def process_posts(
        self, posts: List[Post], window_start: datetime, window_end: datetime, verbose: bool = False
    ) -> List[Post]:
        """
        Полная обработка постов: нормализация, дедупликация, фильтрация.
        
        Args:
            posts: Список постов
            window_start: Начало временного окна
            window_end: Конец временного окна
            verbose: Выводить отладочную информацию
        
        Returns:
            Обработанные посты
        """
        if verbose:
            print(f"      Обработка {len(posts)} постов")
        
        # Нормализуем текст
        for post in posts:
            post.text = self.normalize_text(post.text)
        
        # Дедупликация
        posts_before_dedup = len(posts)
        posts = self.deduplicate_posts(posts)
        if verbose:
            print(f"      После дедупликации: {len(posts)} постов (было {posts_before_dedup})")
        
        # Фильтрация по времени
        posts = self.filter_by_time_window(posts, window_start, window_end, verbose=verbose)
        
        return posts

