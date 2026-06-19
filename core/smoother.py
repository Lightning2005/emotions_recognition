from collections import deque
from typing import List, Dict, Any, Optional


class EmotionSmoother:
    def __init__(self, buffer_size: int = 10):
        self.emotion_history = deque(maxlen=buffer_size)
        self.last_valid_region: Optional[Dict[str, int]] = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Дефолтное пустое состояние
        self.empty_output = {
            "region": None,
            "dominant_emotion": "Unknown",
            "emotion": {emotion: 0.0 for emotion in self.emotion_labels}
        }
        self.last_valid_output: Dict[str, Any] = self.empty_output.copy()

    def update(self, results: List[Dict[str, Any]], is_skipped: bool = False) -> Dict[str, Any]:
        # 1. Если кадр пропущен оптимизацией
        if is_skipped:
            if not self.emotion_history:
                return self.empty_output
            return self.last_valid_output

        # 2. Если ИИ отработал и лица РЕАЛЬНО нет в кадре — мгновенный сброс
        if not results:
            self.clear()
            return self.empty_output

        # 3. Если лицо успешно найдено
        face = results[0]
        self.last_valid_region = face.get('region')
        raw_emotions = face.get('emotion', {})

        if raw_emotions:
            self.emotion_history.append(raw_emotions)

        # Считаем среднее арифметическое по истории
        smoothed_emotions = self._calculate_average_emotions()
        dominant = max(smoothed_emotions, key=smoothed_emotions.get)

        self.last_valid_output = {
            "region": self.last_valid_region,
            "dominant_emotion": dominant,
            "emotion": smoothed_emotions  # Передаем строго сглаженный словарь
        }
        return self.last_valid_output

    def _calculate_average_emotions(self) -> Dict[str, float]:
        if not self.emotion_history:
            return {emotion: 0.0 for emotion in self.emotion_labels}

        sum_emotions = {emotion: 0.0 for emotion in self.emotion_labels}
        for history_item in self.emotion_history:
            for emotion in self.emotion_labels:
                sum_emotions[emotion] += history_item.get(emotion, 0.0)

        history_len = len(self.emotion_history)
        return {emotion: total / history_len for emotion, total in sum_emotions.items()}

    def clear(self) -> None:
        """Полный сброс истории (пригодится при переключении режимов)."""
        self.emotion_history.clear()
        self.last_valid_region = None
        # Синхронизируем структуру дефолтных пустых эмоций, чтобы GUI не падал при чтении dict.get()
        self.last_valid_output = {
            "region": None,
            "dominant_emotion": "Unknown",
            "emotion": {emotion: 0.0 for emotion in self.emotion_labels}
        }