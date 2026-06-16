# core/smoother.py
from collections import Counter, deque
from typing import List, Dict, Any, Optional


class EmotionSmoother:
    def __init__(self, buffer_size: int = 10):
        # Очередь фиксированного размера для хранения истории эмоций
        self.emotion_history = deque(maxlen=buffer_size)
        # Хранилище для последней удачной позиции лица
        self.last_valid_region: Optional[Dict[str, int]] = None

    def update(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Принимает сырые результаты от нейросети и возвращает сглаженные данные."""
        if not results:
            # Если лицо потерялось, возвращаем последнюю известную позицию и самую частую эмоцию
            dominant = self._get_dominant_emotion()
            return {
                "region": self.last_valid_region if self.last_valid_region else {"x": 0, "y": 0, "w": 0, "h": 0},
                "dominant_emotion": dominant if dominant else "Unknown"
            }

        # Берем первое найденное лицо (для простоты)
        face = results[0]
        self.last_valid_region = face.get('region')

        # Записываем текущую эмоцию в историю
        current_emotion = face.get('dominant_emotion')
        if current_emotion:
            self.emotion_history.append(current_emotion)

        return {
            "region": self.last_valid_region,
            "dominant_emotion": self._get_dominant_emotion()
        }

    def _get_dominant_emotion(self) -> Optional[str]:
        """Находит самую частую эмоцию в истории (мажоритарное голосование)."""
        if not self.emotion_history:
            return None
        # Counter подсчитывает количество повторений каждого элемента
        counter = Counter(self.emotion_history)
        # most_common(1) возвращает список вида [('neutral', 5)]
        return counter.most_common(1)[0][0]