import os
from typing import Union, List, Dict, Any
import cv2
from core.engine import EmotionEngine
from core.smoother import EmotionSmoother  # Импортируем наш новый модуль

# "test_happy.jpg" "test_sad.jpg" "test_surprise.jpg" "test.mp4"


# ==================== CONFIGURATION ====================
SOURCE: Union[int, str] = "test_happy.jpg"
FRAME_SKIP_INTERVAL: int = 4
# =======================================================


class EmotionVisualizer:
    def __init__(self, engine: EmotionEngine, smoother: EmotionSmoother, frame_skip: int = 5):
        self.engine = engine
        self.smoother = smoother  # Сохраняем сглаживатель
        self.window_name = "Emotion Analysis"
        self.frame_skip = frame_skip

    def _draw_predictions(self, frame: cv2.Mat, smoothed_data: Dict[str, Any]) -> None:
        """Отрисовка Bounding Box и текста на основе СГЛАЖЕННЫХ данных."""
        region = smoothed_data.get('region')
        if not region:
            return

        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        emotion = smoothed_data.get('dominant_emotion', 'Unknown')

        # Рисуем рамку лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Добавляем текст с эмоцией
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def process_static_image(self, img_path: str) -> None:
        """Обработка одиночного фото."""
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] Не удалось загрузить изображение по пути: {img_path}")
            return

        results = self.engine.analyze(frame)
        # Для фото сглаживание не нужно, берем сырые данные, если они есть
        if results:
            smoothed_data = {"region": results[0].get('region'), "dominant_emotion": results[0].get('dominant_emotion')}
            self._draw_predictions(frame, smoothed_data)

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video_stream(self, source: Union[int, str]) -> None:
        """Обработка видеопотока со сглаживанием и пропуском кадров."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Не удалось открыть источник видео: {source}")
            return

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Шаг анализа (работает редко, экономит FPS)
                if frame_count % self.frame_skip == 0:
                    raw_results = self.engine.analyze(frame)
                    # Отдаем сырые данные в smoother, он их обрабатывает и усредняет
                    smoothed_data = self.smoother.update(raw_results)
                else:
                    # На пропущенных кадрах просто просим smoother вернуть последнее стабильное состояние
                    smoothed_data = self.smoother.update([])

                # 2. Шаг отрисовки (работает каждый кадр, обеспечивая плавность)
                self._draw_predictions(frame, smoothed_data)

                cv2.imshow(self.window_name, frame)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run(self, source: Union[int, str]) -> None:
        if isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.process_static_image(source)
        else:
            self.process_video_stream(source)


def main():
    engine = EmotionEngine()
    smoother = EmotionSmoother(buffer_size=12)  # Создаем сглаживатель с буфером в 12 кадров

    # Передаем обе зависимости в визуализатор
    visualizer = EmotionVisualizer(engine, smoother, frame_skip=FRAME_SKIP_INTERVAL)
    visualizer.run(SOURCE)


if __name__ == "__main__":
    main()