import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


class DatabaseManager:
    def __init__(self, db_dir: str = "database", db_name: str = "analytics.db"):
        self.db_path = os.path.join(db_dir, db_name)
        # Гарантируем, что папка database существует
        os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Создает подключение к БД.
        Используем timeout, чтобы избежать блокировок при одновременных запросах.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row  # Позволяет обращаться к полям по именам, как к словарям
        return conn

    def _init_db(self):
        """Создает таблицы, если они еще не созданы."""
        create_sessions_table = """
        CREATE TABLE IF NOT EXISTS analytics_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,         -- Дата и время сессии (ISO формат)
            source TEXT NOT NULL,            -- Источник (Камера X или имя видеофайла)
            total_frames INTEGER DEFAULT 0,  -- Всего обработанных кадров с лицами
            dominant_emotion TEXT,           -- Самая частая эмоция за всю сессию
            emotions_summary TEXT,           -- JSON-строка со счетчиками всех эмоций
            timeline_data TEXT,              -- Полный JSON-массив с поминутной/покадровой историей
            created_at TEXT NOT NULL
        );
        """
        with self._get_connection() as conn:
            conn.execute(create_sessions_table)
            conn.commit()

    def save_session(self, report_data: Dict[str, Any]) -> int:
        """
        Сохраняет комплексный отчет сессии (видео/камера) в базу данных.
        Возвращает id созданной записи.
        """
        # Извлекаем данные из структуры отчета, которую мы подготовили в gui/app.py
        timestamp = report_data.get("export_timestamp", datetime.now().isoformat())
        source = report_data.get("source", "Unknown")

        summary = report_data.get("summary", {})
        total_frames = summary.get("total_frames_with_faces", 0)
        dominant_emotion = summary.get("most_dominant_emotion_overall", "Unknown")
        emotions_occurrence = summary.get("emotions_occurrence_count", {})

        timeline = report_data.get("timeline", [])

        # Сериализуем сложные структуры в JSON-строки для хранения в TEXT полях
        emotions_summary_json = json.dumps(emotions_occurrence, ensure_ascii=False)
        timeline_json = json.dumps(timeline, ensure_ascii=False)

        query = """
        INSERT INTO analytics_sessions (
            timestamp, source, total_frames, dominant_emotion, emotions_summary, timeline_data, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """

        now_str = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                timestamp, source, total_frames, dominant_emotion,
                emotions_summary_json, timeline_json, now_str
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Возвращает список всех сессий (краткую информацию для списка истории)."""
        query = """
        SELECT id, timestamp, source, total_frames, dominant_emotion, emotions_summary 
        FROM analytics_sessions 
        ORDER BY id DESC;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()

            result = []
            for row in rows:
                session = dict(row)
                # Десериализуем сводку эмоций обратно в словарь Python
                session["emotions_summary"] = json.loads(session["emotions_summary"])
                result.append(session)
            return result

    def get_session_details(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Возвращает полные данные сессии, включая детальный timeline для отрисовки графиков."""
        query = "SELECT * FROM analytics_sessions WHERE id = ?;"
        with self._get_connection() as conn:
            row = conn.execute(query, (session_id,)).fetchone()
            if row is None:
                return None

            session = dict(row)
            session["emotions_summary"] = json.loads(session["emotions_summary"])
            session["timeline_data"] = json.loads(session["timeline_data"])
            return session

    def delete_session(self, session_id: int):
        """Удаляет сессию из истории."""
        query = "DELETE FROM analytics_sessions WHERE id = ?;"
        with self._get_connection() as conn:
            conn.execute(query, (session_id,))
            conn.commit()