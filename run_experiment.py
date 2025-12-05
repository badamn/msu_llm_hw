#!/usr/bin/env python3
"""Скрипт для запуска полного эксперимента end-to-end."""
import argparse
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import yaml
import pytz

from src.models import Post, PriceLabel, NewsFeature, Prediction
from src.ingest_telegram import TelegramPostsParser
from src.load_moex import MOEXClient
from src.baseline import TextProcessor
from src.llm_features import LLMFeatureExtractor
from src.classifier import DirectionClassifier
from src.evaluate import ModelEvaluator


class NewsAnalysisPipeline:
    """Главный пайплайн для анализа новостей и прогнозирования."""
    
    def __init__(self, config_path: str):
        """
        Инициализация пайплайна из конфигурации.
        
        Args:
            config_path: Путь к YAML конфигурации
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.msk_tz = pytz.timezone("Europe/Moscow")
        
        # Инициализация компонентов
        self._init_components()
    
    def _init_components(self):
        """Инициализирует все компоненты пайплайна."""
        # Telegram парсер
        tg_config = self.config["telegram"]
        self.telegram_parser = TelegramPostsParser(
            tgstat_api_token=tg_config["tgstat_api_token"],
            tgstat_base_url=tg_config["tgstat_base_url"],
            telethon_session_name=tg_config["telethon_session_name"],
            telethon_api_id=tg_config["telethon_api_id"],
            telethon_api_hash=tg_config["telethon_api_hash"],
            whitelist_categories_path=tg_config.get("whitelist_categories_path"),
        )
        
        # MOEX клиент
        self.moex_client = MOEXClient()
        
        # Обработчик текста
        self.text_processor = TextProcessor()
        
        # Классификатор
        classifier_config = self.config.get("classifier", {})
        classifier_method = classifier_config.get("method", "zero-shot")
        self.classifier = DirectionClassifier(method=classifier_method)
        
        # LLM экстрактор признаков - инициализируем для всех методов
        # Для zero-shot и few-shot LLM вызывается напрямую в классификаторе
        # Для logistic и gradient_boosting LLM используется для извлечения признаков
        needs_llm = classifier_method in ["zero-shot", "few-shot", "logistic", "gradient_boosting"]

        if needs_llm:
            llm_config = self.config["llm"]
            self.llm_extractor = LLMFeatureExtractor(
                api_key=llm_config["api_key"],
                base_url=llm_config.get("base_url", "https://api.vsegpt.ru/v1"),
                model=llm_config.get("model", "gpt-3.5-turbo"),
                cache_dir=llm_config.get("cache_dir"),
            )
            self.classifier.set_feature_extractor(self.llm_extractor)
        else:
            self.llm_extractor = None
        
        # Оценщик
        self.evaluator = ModelEvaluator()
    
    def generate_labels(
        self, ticker: str, timestamps: List[datetime], horizon_hours: int
    ) -> List[PriceLabel]:
        """
        Генерирует целевые метки для временных точек.
        
        Args:
            ticker: Тикер акции
            timestamps: Список временных меток
            horizon_hours: Горизонт прогноза в часах
        
        Returns:
            Список меток
        """
        labels = []
        
        for base_time in timestamps:
            base_time = base_time.astimezone(self.msk_tz)
            
            # Получаем время закрытия следующего торгового дня
            target_time = self.moex_client._get_next_trading_day_close(base_time)
            
            base_price = self.moex_client.get_close_price(ticker, base_time)
            target_price = self.moex_client.get_close_price(ticker, target_time)
            
            if base_price is None or target_price is None:
                continue
            
            direction = "up" if target_price > base_price else "down"
            
            label = PriceLabel(
                ticker=ticker,
                timestamp=base_time,
                base_price=base_price,
                target_price=target_price,
                direction=direction,
                horizon_hours=horizon_hours,
            )
            labels.append(label)
        
        return labels
    
    async def collect_news(
        self, channels: List[Dict[str, str]], start_date: str, limit: int
    ) -> List[Post]:
        """
        Собирает новости из Telegram каналов.
        
        Args:
            channels: Список каналов
            start_date: Начальная дата (YYYY-MM-DD)
            limit: Количество постов на канал
        
        Returns:
            Список постов
        """
        output_file = self.config["data"]["telegram_output"]
        posts = await self.telegram_parser.collect_channels_posts(
            channels=channels,
            output_file=output_file,
            min_date=start_date,
            limit=limit,
        )
        return posts
    
    def process_news(
        self, posts: List[Post], window_start: datetime, window_end: datetime, verbose: bool = False
    ) -> List[Post]:
        """
        Обрабатывает новости: нормализация, дедупликация, фильтрация.
        
        Args:
            posts: Список постов
            window_start: Начало окна
            window_end: Конец окна
            verbose: Выводить отладочную информацию
        
        Returns:
            Обработанные посты
        """
        return self.text_processor.process_posts(posts, window_start, window_end, verbose=verbose)
    
    def extract_features_for_ticker(
        self, posts: List[Post], ticker: str, verbose: bool = False
    ) -> List[NewsFeature]:
        """
        Извлекает признаки из новостей для конкретного тикера.
        
        Args:
            posts: Список постов
            ticker: Тикер акции
            verbose: Выводить отладочную информацию
        
        Returns:
            Список признаков
        """
        features = []
        
        # Проверяем, нужен ли LLM для извлечения признаков
        # Для zero-shot и few-shot признаки не нужны (LLM вызывается напрямую в классификаторе)
        # Для logistic и gradient_boosting нужны признаки, извлеченные через LLM
        classifier_method = self.classifier.method

        if classifier_method in ["zero-shot", "few-shot"]:
            # Для этих методов признаки не извлекаются заранее
            # LLM будет вызван напрямую в классификаторе
            if verbose:
                print(f"    Пропуск извлечения признаков для {ticker}: метод {classifier_method} использует LLM напрямую")
            return []

        # Для logistic и gradient_boosting извлекаем признаки через LLM
        if not self.llm_extractor:
            if verbose:
                print(f"    Пропуск извлечения признаков для {ticker}: LLM экстрактор не инициализирован")
            return []
        
        if verbose:
            print(f"    Извлечение признаков для {ticker}: получено {len(posts)} постов")
        
        for post in posts:
            # Проверяем, упоминается ли тикер в посте
            mentioned_tickers = self.text_processor.extract_tickers(post.text)
            
            # Извлекаем признаки для всех постов, независимо от упоминания тикера
            # (можно фильтровать позже, если нужно)
            if verbose and mentioned_tickers:
                print(f"      Пост {post.id}: найдены тикеры {mentioned_tickers}")
            
            # Извлекаем признаки для всех постов
            try:
                feature = self.llm_extractor.extract_features(post, ticker)
                features.append(feature)
                if verbose:
                    print(f"      Пост {post.id}: признаки извлечены")
            except Exception as e:
                if verbose:
                    print(f"      Пост {post.id}: ошибка извлечения признаков - {e}")
                continue
        
        if verbose:
            print(f"    Извлечено {len(features)} признаков для {ticker}")
        
        return features
    
    def prepare_training_data(
        self,
        timestamps: List[datetime],
        posts: List[Post],
        labels: List,
        ticker: str,
        window_hours: int,
        verbose: bool = False,
    ) -> Tuple[List[List], List[str]]:
        """
        Подготавливает данные для обучения ML-модели.

        Args:
            timestamps: Временные метки
            posts: Все посты
            labels: Метки для timestamps
            ticker: Тикер
            window_hours: Размер окна
            verbose: Подробный вывод

        Returns:
            Кортеж (features_list, labels_list)
        """
        features_list = []
        labels_list = []

        # Создаем словарь меток по времени для быстрого поиска
        label_dict = {label.timestamp: label.direction for label in labels}

        for idx, timestamp in enumerate(timestamps):
            timestamp = timestamp.astimezone(self.msk_tz)

            # Определяем окно новостей
            window_end = timestamp
            window_start = timestamp - timedelta(hours=window_hours)

            # Фильтруем посты по окну
            window_posts = self.process_news(posts, window_start, window_end, verbose=False)

            if not window_posts:
                continue

            # Извлекаем признаки
            features = self.extract_features_for_ticker(window_posts, ticker, verbose=False)

            if not features:
                continue

            # Находим метку
            closest_label = None
            min_diff = timedelta.max

            for label in labels:
                diff = abs((timestamp - label.timestamp).total_seconds())
                if diff < min_diff.total_seconds():
                    min_diff = timedelta(seconds=diff)
                    closest_label = label

            if closest_label and min_diff < timedelta(days=1):
                features_list.append(features)
                labels_list.append(closest_label.direction)

            if verbose and (idx + 1) % 20 == 0:
                print(f"    Подготовлено {idx+1}/{len(timestamps)} примеров...")

        return features_list, labels_list

    def make_predictions(
        self,
        timestamps: List[datetime],
        posts: List[Post],
        ticker: str,
        window_hours: int,
        verbose: bool = False,
    ) -> List[Prediction]:
        """
        Делает прогнозы для временных точек.
        
        Args:
            timestamps: Список временных меток для прогноза
            posts: Все собранные посты
            ticker: Тикер акции
            window_hours: Размер окна новостей в часах
            verbose: Выводить отладочную информацию
        
        Returns:
            Список прогнозов
        """
        predictions = []
        
        if verbose:
            print(f"  Создание прогнозов для {ticker}: {len(timestamps)} временных точек, {len(posts)} постов всего")
        
        for idx, timestamp in enumerate(timestamps):
            timestamp = timestamp.astimezone(self.msk_tz)
            
            # Определяем окно новостей
            window_end = timestamp
            window_start = timestamp - timedelta(hours=window_hours)
            
            # Показываем вывод для первых 5 временных точек (idx 0-4) для лучшей отладки
            show_debug = verbose and idx < 5
            
            if show_debug:
                print(f"    Точка {idx+1}/{len(timestamps)}: {timestamp.strftime('%Y-%m-%d %H:%M')}")
                print(f"      Окно: {window_start.strftime('%Y-%m-%d %H:%M')} - {window_end.strftime('%Y-%m-%d %H:%M')}")
            
            # Фильтруем посты по окну
            window_posts = self.process_news(posts, window_start, window_end, verbose=show_debug)
            
            if show_debug:
                print(f"      После фильтрации по окну: {len(window_posts)} постов")
            
            if not window_posts:
                if show_debug:
                    print(f"      Пропуск: нет постов в окне")
                continue
            
            # Извлекаем признаки (только для методов, которые их требуют)
            features = self.extract_features_for_ticker(window_posts, ticker, verbose=show_debug)
            
            # Для zero-shot и few-shot признаки не нужны (LLM вызывается напрямую)
            # Для logistic и gradient_boosting нужны признаки
            if self.classifier.method in ["logistic", "gradient_boosting"]:
                if not features:
                    if show_debug:
                        print(f"      Пропуск: не удалось извлечь признаки")
                    continue

            # Делаем прогноз
            if self.classifier.method in ["zero-shot", "few-shot"] and self.llm_extractor:
                # Для LLM-методов агрегируем предсказания по всем постам в окне
                result = self.llm_extractor.aggregate_window_predictions(window_posts, ticker, method="weighted")
                if show_debug:
                    print(f"      Агрегировано {result.get('predictions_count', 0)} предсказаний из {len(window_posts)} постов")
            else:
                # Для ML-методов используем классификатор
                result = self.classifier.predict_with_confidence(features, None, ticker)
            
            prediction = Prediction(
                ticker=ticker,
                timestamp=timestamp,
                news_window_start=window_start,
                news_window_end=window_end,
                predicted_direction=result["direction"],
                confidence=result.get("confidence", 0.5),
                features=features,
                method=self.classifier.method,
            )
            predictions.append(prediction)
            
            if show_debug:
                print(f"      Прогноз создан: {result['direction']} (confidence: {result.get('confidence', 0.5):.2f})")
            
            # Показываем прогресс для всех точек
            if verbose and (idx + 1) % 10 == 0:
                print(f"    Обработано {idx+1}/{len(timestamps)} точек...")
        
        return predictions
    
    def _save_predictions_log(self, all_predictions: Dict, all_labels: Dict, results_path: str):
        """
        Сохраняет детальный лог предсказаний для воспроизводимости.

        Args:
            all_predictions: Словарь предсказаний по тикерам
            all_labels: Словарь меток по тикерам
            results_path: Путь к файлу результатов
        """
        log_path = Path(results_path).parent / "predictions_log.json"

        predictions_log = {
            "timestamp": datetime.now(self.msk_tz).isoformat(),
            "config": {
                "method": self.classifier.method,
                "window_hours": self.config["task"]["window_hours"],
                "horizon_hours": self.config["task"]["horizon_hours"],
                "tickers": self.config["task"]["tickers"],
            },
            "predictions": {},
        }

        for ticker, predictions in all_predictions.items():
            labels = all_labels.get(ticker, [])

            ticker_predictions = []
            for pred in predictions:
                # Находим соответствующую метку
                actual = None
                for label in labels:
                    diff = abs((pred.timestamp - label.timestamp).total_seconds())
                    if diff < 86400:  # < 1 день
                        actual = label.direction
                        break

                ticker_predictions.append({
                    "timestamp": pred.timestamp.isoformat(),
                    "window_start": pred.news_window_start.isoformat(),
                    "window_end": pred.news_window_end.isoformat(),
                    "predicted": pred.predicted_direction,
                    "confidence": pred.confidence,
                    "actual": actual,
                    "correct": pred.predicted_direction == actual if actual else None,
                })

            predictions_log["predictions"][ticker] = ticker_predictions

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(predictions_log, f, ensure_ascii=False, indent=2)

        print(f"  Лог предсказаний сохранен в {log_path}")

    async def run(self):
        """
        Запускает полный пайплайн end-to-end.
        """
        print("=" * 60)
        print("ЗАПУСК ПАЙПЛАЙНА АНАЛИЗА НОВОСТЕЙ")
        print("=" * 60)
        
        task_config = self.config["task"]
        tickers = task_config["tickers"]
        window_hours = task_config["window_hours"]
        horizon_hours = task_config["horizon_hours"]
        
        # Вычисляем даты: если не указаны, используем последние 60 дней
        if task_config.get("end_date") is None:
            end_date_dt = datetime.now(self.msk_tz)
        else:
            end_date_dt = datetime.strptime(task_config["end_date"], "%Y-%m-%d").replace(tzinfo=self.msk_tz)
        
        if task_config.get("start_date") is None:
            start_date_dt = end_date_dt - timedelta(days=60)
        else:
            start_date_dt = datetime.strptime(task_config["start_date"], "%Y-%m-%d").replace(tzinfo=self.msk_tz)
        
        start_date = start_date_dt.strftime("%Y-%m-%d")
        end_date = end_date_dt.strftime("%Y-%m-%d")
        
        print(f"\nПараметры задачи:")
        print(f"  Тикеры: {tickers}")
        print(f"  Окно новостей (W): {window_hours} часов")
        print(f"  Горизонт прогноза (H): {horizon_hours} часов")
        print(f"  Период: {start_date} - {end_date} (последние {int((end_date_dt - start_date_dt).total_seconds() / 86400)} дней)")
        
        # Шаг 1: Сбор новостей
        print("\n[1/6] Сбор новостей из Telegram...")
        channels = self.config.get("channels", [])
        if not channels:
            print("  Предупреждение: список каналов пуст")
            print("  Попытка загрузить посты из сохраненного файла...")
            # Пытаемся загрузить из файла
            posts_file = Path(self.config["data"]["telegram_output"])
            if posts_file.exists():
                with open(posts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    posts = [Post(**p) for p in data.get("posts", [])]
                print(f"  Загружено {len(posts)} постов из файла")
                if posts:
                    # Показываем примеры дат
                    sample_dates = [p.date for p in posts[:3]]
                    print(f"  Примеры дат постов: {sample_dates}")
            else:
                print("  Ошибка: нет каналов и нет сохраненных данных")
                print("  Укажите каналы в config.yaml или соберите данные заранее")
                posts = []
        else:
            posts = await self.collect_news(channels, start_date, limit=100)
            print(f"  Собрано {len(posts)} постов")
        
        # Шаг 2: Загрузка котировок
        print("\n[2/6] Загрузка котировок MOEX...")
        start_dt = start_date_dt
        end_dt = end_date_dt
        
        all_labels = {}
        for ticker in tickers:
            ohlcv = self.moex_client.get_ohlcv(ticker, start_dt, end_dt)
            print(f"  Загружено {len(ohlcv)} записей для {ticker}")
        
        # Шаг 3: Генерация временных точек для прогноза
        print("\n[3/6] Генерация временных точек...")
        
        # Определяем фактический период постов
        if posts:
            # Парсим даты постов и находим диапазон
            post_dates = []
            for post in posts:
                try:
                    post_date_str = post.date
                    if "Z" in post_date_str:
                        post_date_str = post_date_str.replace("Z", "+00:00")
                    elif "+" not in post_date_str and "Z" not in post_date_str:
                        post_date_str = post_date_str + "+00:00"
                    post_dt = datetime.fromisoformat(post_date_str)
                    if post_dt.tzinfo is None:
                        post_dt = pytz.utc.localize(post_dt)
                    post_dt = post_dt.astimezone(self.msk_tz)
                    post_dates.append(post_dt)
                except Exception:
                    continue
            
            if post_dates:
                posts_start = min(post_dates)
                posts_end = max(post_dates)
                print(f"  Фактический период постов: {posts_start.strftime('%Y-%m-%d %H:%M')} - {posts_end.strftime('%Y-%m-%d %H:%M')}")
                
                # Генерируем временные точки на основе фактических дат постов
                # Начинаем с конца периода постов и идем назад
                timestamps = []
                current = posts_end
                # Ограничиваем началом периода постов (нужно окно новостей до момента прогноза)
                # Минимальная точка - это начало постов (окно уже будет включать посты)
                min_timestamp = posts_start
                
                # Генерируем точки каждые 6 часов, начиная с конца
                # Но только если в окне есть посты
                candidate_timestamps = []
                while current >= min_timestamp:
                    if self.moex_client.is_trading_session(current):
                        candidate_timestamps.append(current)
                    current -= timedelta(hours=6)
                
                # Сортируем по возрастанию
                candidate_timestamps.sort()
                
                # Фильтруем: оставляем только те точки, где в окне есть посты
                print(f"  Проверка {len(candidate_timestamps)} кандидатов на наличие постов в окне...")
                for ts in candidate_timestamps:
                    window_start = ts - timedelta(hours=window_hours)
                    window_end = ts
                    # Быстрая проверка: есть ли посты в этом окне
                    posts_in_window = [
                        pd for pd in post_dates 
                        if window_start <= pd <= window_end
                    ]
                    if posts_in_window:
                        timestamps.append(ts)
                
                print(f"  Сгенерировано {len(timestamps)} временных точек с постами в окне (из {len(candidate_timestamps)} кандидатов)")
            else:
                print("  Предупреждение: не удалось определить даты постов")
                timestamps = []
        else:
            print("  Предупреждение: нет постов, временные точки не генерируются")
            timestamps = []
        
        # Если временных точек нет, пробуем использовать период из конфигурации
        if not timestamps:
            print("  Использование периода из конфигурации...")
            timestamps = []
            current = start_dt
            while current <= end_dt:
                if self.moex_client.is_trading_session(current):
                    timestamps.append(current)
                current += timedelta(hours=6)
            print(f"  Сгенерировано {len(timestamps)} временных точек из конфигурации")
        
        # Шаг 4: Генерация меток
        print("\n[4/6] Генерация целевых меток...")
        for ticker in tickers:
            labels = self.generate_labels(ticker, timestamps, horizon_hours)
            all_labels[ticker] = labels
            print(f"  Сгенерировано {len(labels)} меток для {ticker}")
        
        # Шаг 5: Извлечение признаков и прогнозы
        print("\n[5/6] Извлечение признаков и прогнозы...")
        print(f"  Всего постов для анализа: {len(posts)}")
        print(f"  Временных точек: {len(timestamps)}")
        print(f"  Метод классификации: {self.classifier.method}")

        all_predictions = {}

        # Для ML-методов нужен train/test split
        classifier_method = self.classifier.method
        is_ml_method = classifier_method in ["logistic", "gradient_boosting"]

        if is_ml_method and timestamps:
            # Сортируем временные точки по времени
            timestamps_sorted = sorted(timestamps)

            # Train/test split по времени (80/20)
            split_idx = int(len(timestamps_sorted) * 0.8)
            train_timestamps = timestamps_sorted[:split_idx]
            test_timestamps = timestamps_sorted[split_idx:]

            print(f"  Train/test split: {len(train_timestamps)} train / {len(test_timestamps)} test")

            for ticker in tickers:
                print(f"\n  Обработка тикера {ticker}...")
                labels = all_labels.get(ticker, [])

                # Подготовка обучающих данных
                print(f"    Извлечение признаков для обучения ({len(train_timestamps)} точек)...")
                train_features, train_labels = self.prepare_training_data(
                    train_timestamps, posts, labels, ticker, window_hours, verbose=True
                )

                if len(train_features) < 10:
                    print(f"    Предупреждение: недостаточно обучающих данных ({len(train_features)} < 10)")
                    all_predictions[ticker] = []
                    continue

                print(f"    Обучение модели на {len(train_features)} примерах...")
                self.classifier.train(train_features, train_labels, model_type=classifier_method)
                print(f"    Модель обучена!")

                # Прогнозы на тестовых данных
                print(f"    Создание прогнозов на тестовых данных ({len(test_timestamps)} точек)...")
                predictions = self.make_predictions(test_timestamps, posts, ticker, window_hours, verbose=True)
                all_predictions[ticker] = predictions
                print(f"  Сделано {len(predictions)} прогнозов для {ticker}")
        else:
            # Для LLM-методов используем все данные
            for ticker in tickers:
                print(f"\n  Обработка тикера {ticker}...")
                predictions = self.make_predictions(timestamps, posts, ticker, window_hours, verbose=True)
                all_predictions[ticker] = predictions
                print(f"  Сделано {len(predictions)} прогнозов для {ticker}")
        
        # Шаг 6: Оценка качества
        print("\n[6/6] Оценка качества...")
        total_evaluated = 0
        
        for ticker in tickers:
            predictions = all_predictions.get(ticker, [])
            labels = all_labels.get(ticker, [])
            
            if not predictions or not labels:
                print(f"  Пропуск {ticker}: нет прогнозов или меток")
                continue
            
            # Сопоставляем прогнозы и метки по времени
            for pred in predictions:
                # Находим ближайшую метку
                closest_label = None
                min_diff = timedelta.max
                
                for label in labels:
                    diff = abs((pred.timestamp - label.timestamp).total_seconds())
                    if diff < min_diff.total_seconds():
                        min_diff = timedelta(seconds=diff)
                        closest_label = label
                
                # Используем метку только если разница во времени не слишком большая (например, < 1 день)
                if closest_label and min_diff < timedelta(days=1):
                    self.evaluator.add_prediction(
                        pred.predicted_direction, closest_label.direction
                    )
                    total_evaluated += 1
        
        if total_evaluated == 0:
            print("  Предупреждение: нет данных для оценки (нет сопоставленных прогнозов и меток)")
        else:
            print(f"  Оценено {total_evaluated} примеров")
            # Выводим результаты
            self.evaluator.print_results()

            # Сохраняем результаты
            results_path = self.config["output"]["results_path"]
            Path(results_path).parent.mkdir(parents=True, exist_ok=True)
            self.evaluator.save_results(results_path)

            # Сохраняем детальный лог предсказаний для воспроизводимости
            self._save_predictions_log(all_predictions, all_labels, results_path)
        
        print("\n" + "=" * 60)
        print("ПАЙПЛАЙН ЗАВЕРШЕН")
        print("=" * 60)


def main():
    """Главная функция запуска эксперимента."""
    parser = argparse.ArgumentParser(
        description="Запуск эксперимента по анализу новостей Telegram для прогноза MOEX"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Путь к конфигурационному файлу (по умолчанию: configs/config.yaml)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Один тикер для анализа (переопределяет config)",
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Окно новостей в часах (переопределяет config)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Горизонт прогноза в часах (переопределяет config)",
    )
    
    args = parser.parse_args()
    
    # Проверяем существование конфига
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Ошибка: файл конфигурации не найден: {config_path}")
        print("Создайте файл configs/config.yaml или укажите путь через --config")
        sys.exit(1)
    
    # Инициализируем пайплайн
    try:
        pipeline = NewsAnalysisPipeline(str(config_path))
        
        # Переопределяем параметры из аргументов командной строки
        if args.ticker:
            pipeline.config["task"]["tickers"] = [args.ticker]
        if args.window:
            pipeline.config["task"]["window_hours"] = args.window
        if args.horizon:
            pipeline.config["task"]["horizon_hours"] = args.horizon
        
        # Запускаем пайплайн
        asyncio.run(pipeline.run())
        
    except KeyboardInterrupt:
        print("\n\nЭксперимент прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\nОшибка при выполнении эксперимента: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
