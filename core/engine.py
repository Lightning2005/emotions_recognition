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
        # Порог уверенности: если детектор уверен в лице меньше чем на 70%, игнорируем
        self.confidence_threshold = 0.7

    def analyze(self, frame: cv2.Mat) -> List[Dict[str, Any]]:
        """
        Анализирует кадр, фильтрует ложные срабатывания (галлюцинации)
        и гарантированно возвращает СПИСОК реальных результатов.
        """
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False  # Оставляем False, чтобы не падать по Exception
            )

            # Нормализация ответа (dict -> list)
            if not isinstance(results, list):
                results = [results] if results else []

            valid_results = []

            for res in results:
                # Проверяем уверенность детектора в том, что это реально лицо
                # Если параметр отсутствует (в старых версиях), по дефолту берем 1.0
                confidence = res.get('face_confidence', 1.0)

                if confidence >= self.confidence_threshold:
                    valid_results.append(res)

            return valid_results

        except Exception as e:
            print(f"[ERROR] Ошибка во время работы нейросети: {e}")
            return []