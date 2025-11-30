"""Модуль для извлечения признаков через LLM."""
import os
import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
from openai import OpenAI

from src.models import Post, NewsFeature


class LLMFeatureExtractor:
    """Извлекает признаки из новостей через LLM API."""
    
    # JSON-схемы для разных типов запросов
    FEATURES_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "news_features",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"]
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    },
                    "event_type": {
                        "type": "string"
                    },
                    "mentions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "direction_prediction": {
                        "type": "string",
                        "enum": ["up", "down", "neutral"]
                    }
                },
                "required": ["sentiment", "urgency", "event_type", "mentions", "direction_prediction"],
                "additionalProperties": False
            }
        }
    }
    
    PREDICTION_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "price_prediction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"]
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["direction", "confidence"],
                "additionalProperties": False
            }
        }
    }
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.vsegpt.ru/v1",
        model: str = "openai/gpt-5-nano",
        cache_dir: Optional[str] = None,
    ):
        """
        Инициализация клиента LLM.
        
        Args:
            api_key: API ключ для vsegpt.ru
            base_url: Base URL API
            model: Название модели
            cache_dir: Директория для кэширования ответов
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, text_hash: str) -> Optional[Path]:
        """Получает путь к файлу кэша."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{text_hash}.json"
    
    def _load_from_cache(self, text_hash: str) -> Optional[Dict[str, Any]]:
        """Загружает ответ из кэша."""
        cache_path = self._get_cache_path(text_hash)
        if cache_path and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, text_hash: str, response: Dict[str, Any]):
        """Сохраняет ответ в кэш."""
        cache_path = self._get_cache_path(text_hash)
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
    
    def _call_llm(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        max_retries: int = 3,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Вызывает LLM API с обработкой ошибок и retry.
        
        Args:
            prompt: Промпт для модели
            system_prompt: Системный промпт
            max_retries: Максимальное количество попыток
            response_format: Формат ответа (JSON schema или json_object)
        
        Returns:
            Ответ модели
        """
        import hashlib
        # Включаем response_format в хэш для кэширования
        # Сериализуем response_format в строку для хэширования
        response_format_str = json.dumps(response_format, sort_keys=True) if response_format else ""
        cache_key = f"{prompt}_{response_format_str}"
        text_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Проверяем кэш
        cached = self._load_from_cache(text_hash)
        if cached:
            cached_response = cached.get("response", "")
            if cached_response:
                return cached_response
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Подготавливаем параметры запроса
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Низкая температура для более детерминированных ответов
        }
        
        # Добавляем response_format если указан
        # Если API не поддерживает json_schema, можно использовать {"type": "json_object"}
        if response_format:
            request_params["response_format"] = response_format
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**request_params)
                
                if not response or not response.choices:
                    raise ValueError("Пустой ответ от API")
                
                result = response.choices[0].message.content
                
                if not result or not result.strip():
                    raise ValueError("Пустое содержимое ответа")
                
                # Сохраняем в кэш
                self._save_to_cache(text_hash, {"response": result, "prompt": prompt})
                
                return result
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    # Логируем ошибку перед последней попыткой
                    print(f"Ошибка при вызове LLM API (попытка {attempt + 1}/{max_retries}): {e}")
                    print(f"Модель: {self.model}, Base URL: {self.client.base_url}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # Если дошли сюда, значит все попытки не удались
        if last_error:
            raise last_error
        raise RuntimeError("Не удалось получить ответ от LLM")
    
    def extract_features(self, post: Post, ticker: Optional[str] = None) -> NewsFeature:
        """
        Извлекает признаки из поста через LLM.
        
        Args:
            post: Пост из Telegram
            ticker: Тикер, для которого извлекаются признаки
        
        Returns:
            NewsFeature с извлеченными признаками
        """
        system_prompt = """Ты аналитик финансовых новостей. Твоя задача - анализировать новости 
о российских акциях и извлекать релевантные признаки для прогнозирования движения цены."""
        
        prompt = f"""Проанализируй следующую новость и извлеки признаки:

Новость:
{post.text}

Извлеки следующую информацию в формате JSON:
{{
    "sentiment": "positive/negative/neutral",
    "urgency": "high/medium/low",
    "event_type": "корпоративные новости/макроэкономика/регуляторные/другое",
    "mentions": ["список упоминаний тикеров или компаний"],
    "direction_prediction": "up/down/neutral"
}}

Если новость не относится к указанному тикеру ({ticker if ticker else 'любому'}), 
укажи direction_prediction как "neutral".

Ответь только JSON, без дополнительного текста."""
        
        try:
            # Используем JSON-схему для гарантии правильного формата
            response = self._call_llm(
                prompt, 
                system_prompt, 
                response_format=self.FEATURES_SCHEMA
            )
            
            if not response or not response.strip():
                raise ValueError("Пустой ответ от LLM")
            
            # Парсим JSON ответ
            # Убираем возможные markdown блоки
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            features_dict = json.loads(response)
            
            return NewsFeature(
                post_id=post.id,
                ticker=ticker,
                sentiment=features_dict.get("sentiment"),
                urgency=features_dict.get("urgency"),
                event_type=features_dict.get("event_type"),
                mentions=features_dict.get("mentions", []),
                direction_prediction=features_dict.get("direction_prediction"),
                raw_features=features_dict,
            )
        except json.JSONDecodeError as e:
            # Логируем ошибку парсинга JSON
            print(f"Ошибка парсинга JSON в extract_features для поста {post.id}: {e}")
            print(f"Ответ LLM: {response if 'response' in locals() else 'не получен'}")
            return NewsFeature(
                post_id=post.id,
                ticker=ticker,
                sentiment=None,
                urgency=None,
                event_type=None,
                mentions=[],
                direction_prediction=None,
                raw_features={"error": f"JSON decode error: {str(e)}", "raw_response": response if 'response' in locals() else None},
            )
        except Exception as e:
            # В случае ошибки возвращаем пустые признаки
            print(f"Ошибка в extract_features для поста {post.id}: {e}")
            return NewsFeature(
                post_id=post.id,
                ticker=ticker,
                sentiment=None,
                urgency=None,
                event_type=None,
                mentions=[],
                direction_prediction=None,
                raw_features={"error": str(e)},
            )
    
    def zero_shot_classify(self, post: Post, ticker: str) -> str:
        """
        Zero-shot классификация направления изменения цены.
        
        Args:
            post: Пост из Telegram
            ticker: Тикер акции
        
        Returns:
            "up" или "down"
        """
        result = self.zero_shot_classify_with_confidence(post, ticker)
        return result["direction"]
    
    def zero_shot_classify_with_confidence(self, post: Post, ticker: str) -> Dict[str, Any]:
        """
        Zero-shot классификация с оценкой уверенности.
        
        Args:
            post: Пост из Telegram
            ticker: Тикер акции
        
        Returns:
            Словарь с direction и confidence
        """
        system_prompt = """Ты финансовый аналитик. Твоя задача - предсказать направление 
изменения цены акции на основе новости. Отвечай в формате JSON с полями "direction" и "confidence"."""
        
        prompt = f"""На основе следующей новости предскажи направление изменения цены акции {ticker}:

{post.text}

Ответь в формате JSON:
{{
    "direction": "up" или "down",
    "confidence": число от 0 до 1 (уверенность в предсказании)
}}

Ответь только JSON, без дополнительного текста."""
        
        try:
            # Используем JSON-схему для гарантии правильного формата
            response = self._call_llm(
                prompt, 
                system_prompt,
                response_format=self.PREDICTION_SCHEMA
            )
            
            if not response or not response.strip():
                raise ValueError("Пустой ответ от LLM")
            
            # Парсим JSON ответ (схема должна гарантировать правильный формат)
            response = response.strip()
            # Убираем возможные markdown блоки на всякий случай
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            result = json.loads(response)
            direction = result.get("direction", "up").lower()
            confidence = float(result.get("confidence", 0.5))
            
            # Нормализуем direction (на всякий случай, хотя схема должна гарантировать)
            if direction not in ["up", "down"]:
                direction = "up"
            
            # Ограничиваем confidence (на всякий случай, хотя схема должна гарантировать)
            confidence = max(0.0, min(1.0, confidence))
            
            return {"direction": direction, "confidence": confidence}
        except Exception as e:
            # Логируем ошибку для отладки
            print(f"Ошибка в zero_shot_classify_with_confidence: {e}")
            print(f"Ответ LLM: {response if 'response' in locals() else 'не получен'}")
            return {"direction": "up", "confidence": 0.5}
    
    def few_shot_classify(
        self, post: Post, ticker: str, examples: List[Dict[str, str]]
    ) -> str:
        """
        Few-shot классификация с примерами.
        
        Args:
            post: Пост из Telegram
            ticker: Тикер акции
            examples: Список примеров [{"text": "...", "direction": "up/down"}]
        
        Returns:
            "up" или "down"
        """
        result = self.few_shot_classify_with_confidence(post, ticker, examples)
        return result["direction"]
    
    def few_shot_classify_with_confidence(
        self, post: Post, ticker: str, examples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Few-shot классификация с оценкой уверенности.
        
        Args:
            post: Пост из Telegram
            ticker: Тикер акции
            examples: Список примеров [{"text": "...", "direction": "up/down"}]
        
        Returns:
            Словарь с direction и confidence
        """
        system_prompt = """Ты финансовый аналитик. Твоя задача - предсказать направление 
изменения цены акции на основе новости. Отвечай в формате JSON с полями "direction" и "confidence"."""
        
        examples_text = "\n\n".join(
            [
                f"Новость: {ex['text']}\nНаправление: {ex['direction']}"
                for ex in examples
            ]
        )
        
        prompt = f"""Примеры:

{examples_text}

Новость для анализа:
{post.text}

Предскажи направление изменения цены акции {ticker} в формате JSON:
{{
    "direction": "up" или "down",
    "confidence": число от 0 до 1 (уверенность в предсказании)
}}

Ответь только JSON, без дополнительного текста."""
        
        try:
            # Используем JSON-схему для гарантии правильного формата
            response = self._call_llm(
                prompt, 
                system_prompt,
                response_format=self.PREDICTION_SCHEMA
            )
            
            if not response or not response.strip():
                raise ValueError("Пустой ответ от LLM")
            
            # Парсим JSON ответ (схема должна гарантировать правильный формат)
            response = response.strip()
            # Убираем возможные markdown блоки на всякий случай
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            result = json.loads(response)
            direction = result.get("direction", "up").lower()
            confidence = float(result.get("confidence", 0.5))
            
            # Нормализуем direction (на всякий случай, хотя схема должна гарантировать)
            if direction not in ["up", "down"]:
                direction = "up"
            
            # Ограничиваем confidence (на всякий случай, хотя схема должна гарантировать)
            confidence = max(0.0, min(1.0, confidence))
            
            return {"direction": direction, "confidence": confidence}
        except Exception as e:
            # Логируем ошибку для отладки
            print(f"Ошибка в few_shot_classify_with_confidence: {e}")
            print(f"Ответ LLM: {response if 'response' in locals() else 'не получен'}")
            return {"direction": "up", "confidence": 0.5}
    
    def aggregate_features(
        self, features: List[NewsFeature], method: str = "majority"
    ) -> Dict[str, Any]:
        """
        Агрегирует признаки за окно W.
        
        Args:
            features: Список признаков
            method: Метод агрегации ("majority", "weighted")
        
        Returns:
            Агрегированные признаки
        """
        if not features:
            return {"direction": "up", "confidence": 0.5}
        
        if method == "majority":
            directions = [f.direction_prediction for f in features if f.direction_prediction]
            if not directions:
                return {"direction": "up", "confidence": 0.5}
            
            up_count = directions.count("up")
            down_count = directions.count("down")
            
            if up_count > down_count:
                direction = "up"
                confidence = up_count / len(directions)
            elif down_count > up_count:
                direction = "down"
                confidence = down_count / len(directions)
            else:
                direction = "up"
                confidence = 0.5
            
            return {
                "direction": direction,
                "confidence": confidence,
                "total_news": len(features),
                "up_count": up_count,
                "down_count": down_count,
            }
        
        return {"direction": "up", "confidence": 0.5}

