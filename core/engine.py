from typing import List, Dict, Any
from deepface import DeepFace
import cv2


class EmotionEngine:
    def __init__(self, detector_backend: str = 'ssd'):
        """
        Инициализация аналитического движка.
        Используем 'ssd' — он быстрый на CPU и отлично держит углы обзора.
        """
        self.detector_backend = detector_backend

    def analyze(self, frame: cv2.Mat) -> List[Dict[str, Any]]:
        """
        Анализирует кадр и гарантированно возвращает СПИСОК результатов.
        """
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False  # Предотвращает падение при потере лица
            )

            # Нормализация ответа: старые версии DeepFace возвращают dict, новые — list.
            # Как Senior, мы обязаны подстраховаться, чтобы код не упал.
            if isinstance(results, list):
                return results
            return [results] if results else []

        except Exception as e:
            print(f"[ERROR] Ошибка во время работы нейросети: {e}")
            return []