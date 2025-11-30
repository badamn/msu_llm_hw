"""Модуль для классификации направления изменения цены."""
from typing import List, Dict, Optional, Any
from datetime import datetime
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

from src.models import NewsFeature, Prediction


class DirectionClassifier:
    """Классификатор направления изменения цены."""
    
    def __init__(self, method: str = "zero-shot"):
        """
        Инициализация классификатора.
        
        Args:
            method: Метод классификации ("zero-shot", "few-shot", "logistic", "gradient_boosting")
        """
        self.method = method
        self.model = None
        self.feature_extractor = None
    
    def set_feature_extractor(self, feature_extractor):
        """Устанавливает экстрактор признаков для LLM методов."""
        self.feature_extractor = feature_extractor
    
    def _extract_numerical_features(self, features: List[NewsFeature]) -> np.ndarray:
        """
        Извлекает численные признаки из NewsFeature для обучения модели.
        
        Args:
            features: Список признаков
        
        Returns:
            Массив численных признаков
        """
        feature_vectors = []
        
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        urgency_map = {"high": 2, "medium": 1, "low": 0}
        
        for feat in features:
            vector = [
                sentiment_map.get(feat.sentiment, 0) if feat.sentiment else 0,
                urgency_map.get(feat.urgency, 0) if feat.urgency else 0,
                1 if feat.direction_prediction == "up" else 0,
                1 if feat.direction_prediction == "down" else 0,
                len(feat.mentions),
            ]
            feature_vectors.append(vector)
        
        return np.array(feature_vectors)
    
    def train(
        self,
        features_list: List[List[NewsFeature]],
        labels: List[str],
        model_type: str = "logistic",
    ):
        """
        Обучает легкую модель поверх признаков.
        
        Args:
            features_list: Список списков признаков для каждого примера
            labels: Список меток ("up" или "down")
            model_type: Тип модели ("logistic" или "gradient_boosting")
        """
        if model_type == "logistic":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Агрегируем признаки для каждого примера
        aggregated_features = []
        for features in features_list:
            if features:
                # Берем среднее по всем признакам
                feat_array = self._extract_numerical_features(features)
                aggregated = feat_array.mean(axis=0)
            else:
                aggregated = np.zeros(5)
            aggregated_features.append(aggregated)
        
        X = np.array(aggregated_features)
        y = np.array([1 if label == "up" else 0 for label in labels])
        
        self.model.fit(X, y)
        self.method = model_type
    
    def predict(
        self, features: List[NewsFeature], post: Optional[Any] = None, ticker: Optional[str] = None
    ) -> str:
        """
        Предсказывает направление на основе признаков.
        
        Args:
            features: Список признаков новостей
            post: Пост (для zero-shot/few-shot методов)
            ticker: Тикер (для zero-shot/few-shot методов)
        
        Returns:
            "up" или "down"
        """
        if self.method == "zero-shot":
            if not self.feature_extractor or not post or not ticker:
                return "up"
            return self.feature_extractor.zero_shot_classify(post, ticker)
        
        elif self.method == "few-shot":
            if not self.feature_extractor or not post or not ticker:
                return "up"
            # Для few-shot нужны примеры, упрощенная версия
            return self.feature_extractor.zero_shot_classify(post, ticker)
        
        elif self.method in ["logistic", "gradient_boosting"]:
            if self.model is None:
                return "up"
            
            if not features:
                return "up"
            
            feat_array = self._extract_numerical_features(features)
            aggregated = feat_array.mean(axis=0).reshape(1, -1)
            
            prediction = self.model.predict(aggregated)[0]
            return "up" if prediction == 1 else "down"
        
        else:
            # Fallback: агрегация через LLM
            if features:
                aggregated = self.feature_extractor.aggregate_features(features)
                return aggregated.get("direction", "up")
            return "up"
    
    def predict_with_confidence(
        self, features: List[NewsFeature], post: Optional[Any] = None, ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Предсказывает направление с оценкой уверенности.
        
        Args:
            features: Список признаков
            post: Пост (для LLM методов)
            ticker: Тикер
        
        Returns:
            Словарь с direction и confidence
        """
        if self.method == "zero-shot":
            # Для zero-shot вызываем LLM напрямую
            if not self.feature_extractor or not post or not ticker:
                return {"direction": "up", "confidence": 0.5}
            return self.feature_extractor.zero_shot_classify_with_confidence(post, ticker)
        
        elif self.method == "few-shot":
            # Для few-shot вызываем LLM напрямую
            if not self.feature_extractor or not post or not ticker:
                return {"direction": "up", "confidence": 0.5}
            # Для few-shot нужны примеры, упрощенная версия - используем zero-shot
            return self.feature_extractor.zero_shot_classify_with_confidence(post, ticker)
        
        elif self.method in ["logistic", "gradient_boosting"]:
            if self.model is None or not features:
                return {"direction": "up", "confidence": 0.5}
            
            feat_array = self._extract_numerical_features(features)
            aggregated = feat_array.mean(axis=0).reshape(1, -1)
            
            proba = self.model.predict_proba(aggregated)[0]
            direction = "up" if proba[1] > 0.5 else "down"
            confidence = max(proba)
            
            return {"direction": direction, "confidence": float(confidence)}
        
        else:
            # Fallback: агрегация через LLM (для других методов)
            if self.feature_extractor and features:
                aggregated = self.feature_extractor.aggregate_features(features)
                return aggregated
            return {"direction": "up", "confidence": 0.5}
    
    def save(self, path: str):
        """Сохраняет модель в файл."""
        if self.model is not None:
            with open(path, "wb") as f:
                pickle.dump({"model": self.model, "method": self.method}, f)
    
    def load(self, path: str):
        """Загружает модель из файла."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.method = data["method"]

