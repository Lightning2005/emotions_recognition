import threading
import time
from queue import Queue
from typing import Union
import cv2
from core.engine import EmotionEngine
from core.smoother import EmotionSmoother


class VideoWorker:
    def __init__(self, source: Union[int, str], engine: EmotionEngine, smoother: EmotionSmoother, frame_skip: int = 4):
        self.source = source
        self.engine = engine
        self.smoother = smoother
        self.frame_skip = frame_skip

        self.result_queue = Queue(maxsize=2)

        self._is_running = False
        self._is_paused = False
        self._seek_target: Union[int, None] = None  # Целевой сдвиг в секундах
        self._rewind_to_start = False               # Флаг полного сброса на 0
        self._lock = threading.Lock()
        self._thread: Union[threading.Thread, None] = None

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._is_paused = False
        self._seek_target = None
        self._rewind_to_start = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._is_paused = paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._is_paused

    def seek_relative(self, seconds: float) -> None:
        """Запрос на относительную перемотку (в секундах)."""
        with self._lock:
            if isinstance(self.source, str):
                self._seek_target = int(seconds)

    def rewind_to_start(self) -> None:
        """Безопасный мгновенный сброс видео на начало."""
        with self._lock:
            if isinstance(self.source, str):
                self._rewind_to_start = True
                self._is_paused = False  # Снимаем с паузы при повторе

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Поток не смог открыть источник: {self.source}")
            self._is_running = False
            return

        is_file = isinstance(self.source, str)
        file_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 30
        if file_fps <= 0:
            file_fps = 30
        frame_delay = 1.0 / file_fps

        frame_count = 0

        while self._is_running:
            start_time = time.time()

            # Обработка команд управления под Lock
            with self._lock:
                # 1. Проверяем флаг полного сброса на начало
                if self._rewind_to_start and is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    self.smoother.clear()
                    self._rewind_to_start = False
                    self._seek_target = None  # Сбрасываем другие запросы перемотки, если они были

                # 2. Относительная перемотка (+/- 5 секунд)
                elif self._seek_target is not None and is_file:
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    frame_delta = self._seek_target * file_fps
                    target_frame = max(0, int(current_frame + frame_delta))

                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    frame_count = target_frame
                    self.smoother.clear()
                    self._seek_target = None

                # 3. Пауза
                if self._is_paused:
                    time.sleep(0.05)
                    continue

            ret, frame = cap.read()
            if not ret:
                # Если видео закончилось, не выходим из цикла насовсем,
                # чтобы пользователь мог нажать "Повтор" или "Перемотка назад"
                if is_file:
                    time.sleep(0.1)
                    continue
                else:
                    print("[INFO] Поток камеры остановлен.")
                    break

            if frame_count % self.frame_skip == 0:
                raw_results = self.engine.analyze(frame)
                smoothed_data = self.smoother.update(raw_results, is_skipped=False)
            else:
                smoothed_data = self.smoother.update([], is_skipped=True)

            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except Exception:
                    pass

            # Собираем метаданные для плеера
            meta = {
                "current_frame": frame_count,
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0,
                "fps": file_fps
            }
            self.result_queue.put((frame, smoothed_data, meta))
            frame_count += 1

            if is_file:
                elapsed = time.time() - start_time
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.01)

        cap.release()
        self._is_running = False

class VideoExportWorker:
    def __init__(self, source_path: str, output_path: str, engine: EmotionEngine, limit_frames: int = None):
        self.source_path = source_path
        self.output_path = output_path
        self.engine = engine
        self.limit_frames = limit_frames  # None — всё видео, иначе число кадров

        self._is_running = False
        self.progress = 0.0
        self._thread = None
        self.on_progress_callback = None
        self.on_finished_callback = None

    def start(self, on_progress, on_finished) -> None:
        self.on_progress_callback = on_progress
        self.on_finished_callback = on_finished
        self._is_running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            self._is_running = False
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0
        if total_frames <= 0:
            total_frames = 1

        # Если выбран экспорт до текущего момента, пересчитываем лимит
        target_total_frames = total_frames
        if self.limit_frames is not None and self.limit_frames > 0:
            target_total_frames = min(self.limit_frames, total_frames)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        current_frame = 0

        while self._is_running and current_frame < target_total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            raw_results = self.engine.analyze(frame)

            if raw_results and len(raw_results) > 0:
                result = raw_results[0]
                dominant_emotion = result.get('dominant_emotion', 'Unknown')
                region = result.get('region', {})

                if dominant_emotion and dominant_emotion != "Unknown" and region:
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, dominant_emotion.upper(), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            out.write(frame)
            current_frame += 1

            # Прогресс считаем от целевого объема кадров
            self.progress = current_frame / target_total_frames

            if self.on_progress_callback:
                self.on_progress_callback(self.progress)

        cap.release()
        out.release()
        self._is_running = False

        if self.on_finished_callback:
            self.on_finished_callback()