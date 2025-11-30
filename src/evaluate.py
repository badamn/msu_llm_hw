"""Модуль для оценки качества модели."""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import json
from pathlib import Path


class ModelEvaluator:
    """Оценка качества модели классификации."""
    
    def __init__(self):
        self.predictions: List[str] = []
        self.labels: List[str] = []
    
    def add_prediction(self, prediction: str, label: str):
        """
        Добавляет предсказание и метку.
        
        Args:
            prediction: Предсказание ("up" или "down")
            label: Истинная метка ("up" или "down")
        """
        self.predictions.append(prediction)
        self.labels.append(label)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Вычисляет метрики качества.
        
        Returns:
            Словарь с метриками
        """
        if not self.predictions or not self.labels:
            return {}
        
        # Конвертируем в числовой формат для sklearn
        y_pred = [1 if p == "up" else 0 for p in self.predictions]
        y_true = [1 if l == "up" else 0 for l in self.labels]
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Получает confusion matrix.
        
        Returns:
            Массив confusion matrix
        """
        if not self.predictions or not self.labels:
            return np.array([[0, 0], [0, 0]])
        
        y_pred = [1 if p == "up" else 0 for p in self.predictions]
        y_true = [1 if l == "up" else 0 for l in self.labels]
        
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self) -> str:
        """
        Получает текстовый отчет о классификации.
        
        Returns:
            Отчет в формате sklearn classification_report
        """
        if not self.predictions or not self.labels:
            return "No predictions available"
        
        y_pred = [1 if p == "up" else 0 for p in self.predictions]
        y_true = [1 if l == "up" else 0 for l in self.labels]
        
        return classification_report(
            y_true, y_pred, target_names=["down", "up"], zero_division=0
        )
    
    def print_results(self):
        """Выводит результаты оценки."""
        metrics = self.calculate_metrics()
        cm = self.get_confusion_matrix()
        report = self.get_classification_report()
        
        print("\n" + "=" * 50)
        print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
        print("=" * 50)
        
        print("\nМетрики:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
        
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Down    Up")
        print(f"Actual Down   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"        Up    {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        print("\nClassification Report:")
        print(report)
        
        print("\nДетализация:")
        print(f"  True Positives:  {metrics.get('true_positives', 0)}")
        print(f"  True Negatives:  {metrics.get('true_negatives', 0)}")
        print(f"  False Positives: {metrics.get('false_positives', 0)}")
        print(f"  False Negatives: {metrics.get('false_negatives', 0)}")
        print("=" * 50 + "\n")
    
    def save_results(self, output_path: str):
        """
        Сохраняет результаты в файл.
        
        Args:
            output_path: Путь к файлу для сохранения
        """
        metrics = self.calculate_metrics()
        cm = self.get_confusion_matrix().tolist()
        report = self.get_classification_report()
        
        results = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
            "total_samples": len(self.predictions),
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Результаты сохранены в {output_path}")
    
    def reset(self):
        """Сбрасывает накопленные предсказания и метки."""
        self.predictions = []
        self.labels = []

