import os
import cv2
import json
from datetime import datetime
from typing import Dict, Any, Union, List


class DataExporter:
    def __init__(self, base_dir: str = "output"):
        self.base_dir = base_dir
        self.snapshots_dir = os.path.join(base_dir, "snapshots")
        self.reports_dir = os.path.join(base_dir, "reports")

        # Гарантируем наличие папок при инициализации
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def _get_timestamp(self) -> str:
        """Генерирует временную метку для уникальности имен файлов."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_snapshot(self, frame: cv2.Mat) -> str:
        """
        Сохраняет текущий кадр (с рамкой и текстом, если они отрисованы на нем) на диск.
        Возвращает путь к сохраненному файлу.
        """
        if frame is None:
            raise ValueError("Кадр пуст, сохранение невозможно.")

        filename = f"snapshot_{self._get_timestamp()}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)

        # Сохраняем картинку средствами OpenCV
        success = cv2.imwrite(filepath, frame)
        if not success:
            raise IOError(f"Не удалось записать файл по пути: {filepath}")

        return filepath

    def save_report(self, report_data: Union[Dict[str, Any], List[Any]]) -> str:
        """
        Сохраняет показатели эмоций в формате JSON.
        Поддерживает как одиночные кадры (фото), так и комплексную историю (видео/камера).
        """
        filename = f"report_{self._get_timestamp()}.json"
        filepath = os.path.join(self.reports_dir, filename)

        # Проверяем, пришел ли уже готовый сформированный отчет для видео
        # (если в словаре есть ключ 'timeline' или 'project_name')
        if isinstance(report_data, dict) and ("timeline" in report_data or "project_name" in report_data):
            final_data = report_data
        else:
            # Если прилетел сырой одиночный кадр (например, с фото), бережно собираем старую структуру
            raw_distribution = report_data.get("emotion", {}) if isinstance(report_data, dict) else {}

            clean_distribution = {}
            for emotion, score in raw_distribution.items():
                clean_distribution[emotion] = float(score)  # Конвертируем float32 -> float

            final_data = {
                "timestamp": datetime.now().isoformat(),
                "dominant_emotion": report_data.get("dominant_emotion", "Unknown") if isinstance(report_data,
                                                                                                 dict) else "Unknown",
                "emotions_distribution": clean_distribution
            }

        # Записываем итоговую структуру в файл
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        return filepath