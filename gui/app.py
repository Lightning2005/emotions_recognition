import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import cv2
from PIL import Image
from tkinter import filedialog
from database.manager import DatabaseManager

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.engine import EmotionEngine
from core.smoother import EmotionSmoother
from core.exporter import DataExporter
from gui.threads import VideoWorker, VideoExportWorker
from gui.widgets import HistoryWidget

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class CustomExportDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Настройка экспорта")
        self.geometry("400x200")
        self.resizable(False, False)

        # Делаем окно модальным (блокирует основное окно, пока не закроют)
        self.lift()
        self.grab_set()

        self.result = None  # Сюда запишем выбор: "all", "part" или None

        self.label = ctk.CTkLabel(self, text="Вы хотите сохранить всё видео целиком?",
                                  font=ctk.CTkFont(size=14, weight="bold"), wraplength=350)
        self.label.pack(pady=(25, 10))

        self.sub_label = ctk.CTkLabel(self, text="Выбор 'Часть' сохранит видео от начала до текущей паузы.",
                                      font=ctk.CTkFont(size=12), text_color="gray", wraplength=350)
        self.sub_label.pack(pady=(0, 20))

        # Фрейм для кнопок
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20)

        btn_all = ctk.CTkButton(btn_frame, text="Всё видео", width=100, fg_color="#1a73e8", command=self.choose_all)
        btn_all.pack(side="left", expand=True, padx=5)

        btn_part = ctk.CTkButton(btn_frame, text="Часть видео", width=100, fg_color="#e67e22", command=self.choose_part)
        btn_part.pack(side="left", expand=True, padx=5)

        btn_cancel = ctk.CTkButton(btn_frame, text="Отмена", width=100, fg_color="#555555", command=self.choose_cancel)
        btn_cancel.pack(side="left", expand=True, padx=5)

    def choose_all(self):
        self.result = "all"
        self.destroy()

    def choose_part(self):
        self.result = "part"
        self.destroy()

    def choose_cancel(self):
        self.result = None
        self.destroy()


class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Emotion Recognition System Pro")
        # Увеличиваем ширину до 1440, чтобы сайдбару и плееру хватало места без сжатия
        self.geometry("1440x860")
        self.minsize(1100, 780)

        self.engine = EmotionEngine(detector_backend='ssd')
        self.smoother = EmotionSmoother(buffer_size=10)
        self.exporter = DataExporter()
        self.db_manager = DatabaseManager()

        self.worker = None
        self.video_source = 0
        self.current_frame_skip = 4
        self.current_video_frame_idx = 0

        self.last_frame = None
        self.last_emotion_data = {}
        self.video_history_data = []

        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Главная сетка окна - вкладки занимают всё доступное пространство
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # ВЕРХНИЙ УРОВЕНЬ: ВКЛАДКИ (TABVIEW)
        # ==========================================
        self.tab_view = ctk.CTkTabview(self, segmented_button_selected_color="#1f538d",
                                       segmented_button_selected_hover_color="#1a73e8")
        # Уменьшаем боковые отступы вкладок, чтобы выиграть полезную ширину
        self.tab_view.grid(row=0, column=0, sticky="nsew", padx=10, pady=(5, 10))

        self.tab_view.add("Анализ")
        self.tab_view.add("История сессий")

        # КРИТИЧЕСКИЙ ФИКС СЕТКИ: жесткое разделение пространства на вкладке "Анализ"
        self.tab_view.tab("Анализ").grid_columnconfigure(0, weight=0)  # Левая панель (не растет)
        self.tab_view.tab("Анализ").grid_columnconfigure(1, weight=1)  # Плеер (забирает ВСЁ оставшееся место)
        self.tab_view.tab("Анализ").grid_rowconfigure(0, weight=1)

        # ==========================================
        # 1. ЛЕВАЯ ПАНЕЛЬ (SIDEBAR)
        # ==========================================
        # Фиксируем ширину сайдбара, чтобы он не раздувался
        self.sidebar_frame = ctk.CTkFrame(self.tab_view.tab("Анализ"), width=260, corner_radius=6)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.sidebar_frame.grid_propagate(False)
        self.sidebar_frame.grid_rowconfigure(14, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Control Panel",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        self.btn_camera_mode = ctk.CTkButton(self.sidebar_frame, text="Режим: Камера", command=self.toggle_camera)
        self.btn_camera_mode.grid(row=1, column=0, padx=20, pady=6, sticky="ew")

        self.btn_load_image = ctk.CTkButton(self.sidebar_frame, text="Открыть фото", fg_color="transparent",
                                            border_width=2, command=self.load_image_action)
        self.btn_load_image.grid(row=2, column=0, padx=20, pady=6, sticky="ew")

        self.btn_load_video = ctk.CTkButton(self.sidebar_frame, text="Открыть видео", fg_color="transparent",
                                            border_width=2, command=self.load_video_action)
        self.btn_load_video.grid(row=3, column=0, padx=20, pady=6, sticky="ew")

        self.separator_1 = ctk.CTkLabel(self.sidebar_frame, text="—" * 20, text_color="gray50")
        self.separator_1.grid(row=4, column=0, padx=20, pady=2)

        self.lbl_camera_select = ctk.CTkLabel(self.sidebar_frame, text="Выбор источника камеры:",
                                              font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_camera_select.grid(row=5, column=0, padx=20, pady=(2, 2), sticky="w")

        self.combo_camera = ctk.CTkComboBox(self.sidebar_frame, values=["Камера 0", "Камера 1", "Камера 2"],
                                            command=self.change_camera_source)
        self.combo_camera.set(f"Камера {self.video_source}")
        self.combo_camera.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.lbl_frameskip = ctk.CTkLabel(self.sidebar_frame, text=f"Пропуск кадров: {self.current_frame_skip}",
                                          font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_frameskip.grid(row=7, column=0, padx=20, pady=(5, 2), sticky="w")

        self.slider_frameskip = ctk.CTkSlider(self.sidebar_frame, from_=1, to=10, number_of_steps=9,
                                              command=self.change_frameskip_value)
        self.slider_frameskip.set(self.current_frame_skip)
        self.slider_frameskip.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.separator_2 = ctk.CTkLabel(self.sidebar_frame, text="—" * 20, text_color="gray50")
        self.separator_2.grid(row=9, column=0, padx=20, pady=2)

        self.btn_snapshot = ctk.CTkButton(self.sidebar_frame, text="Сделать снимок",
                                          fg_color="#2c3e50", hover_color="#34495e",
                                          state="disabled", command=self.save_snapshot_action)
        self.btn_snapshot.grid(row=10, column=0, padx=20, pady=6, sticky="ew")

        self.btn_export_report = ctk.CTkButton(self.sidebar_frame, text="Экспорт аналитики (JSON)",
                                               fg_color="#2c3e50", hover_color="#34495e",
                                               state="disabled", command=self.save_report_action)
        self.btn_export_report.grid(row=11, column=0, padx=20, pady=6, sticky="ew")

        self.btn_export_video = ctk.CTkButton(self.sidebar_frame, text="Сохранить видео",
                                              fg_color="#2c3e50", hover_color="#34495e",
                                              state="disabled", command=self.save_video_action)
        self.btn_export_video.grid(row=12, column=0, padx=20, pady=6, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame, orientation="horizontal", mode="determinate")
        self.progress_bar.set(0)

        self.emotion_status_label = ctk.CTkLabel(self.sidebar_frame, text="Эмоция: ---", font=ctk.CTkFont(size=16))
        self.emotion_status_label.grid(row=15, column=0, padx=20, pady=15, sticky="s")

        # ==========================================
        # 2. ГЛАВНАЯ ПАНЕЛЬ ПЛЕЕРА (MAIN FRAME)
        # ==========================================
        # Убираем огромный отступ слева (padx=(5, 0)), давая фрейму растянуться до сайдбара
        self.main_frame = ctk.CTkFrame(self.tab_view.tab("Анализ"), fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=6)  # Экран видео
        self.main_frame.grid_rowconfigure(1, weight=0)  # Контроллеры плеера
        self.main_frame.grid_rowconfigure(2, weight=4)  # Нижний график (увеличили вес с 3 до 4)

        self.video_label = ctk.CTkLabel(self.main_frame, text="Выберите режим работы в левой панели",
                                        fg_color="#101010", corner_radius=8)
        self.video_label.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        # --- ПАНЕЛЬ МЕДИАПЛЕЕРА ---
        self.media_control_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.media_control_frame.grid(row=1, column=0, sticky="ew", pady=5)
        self.media_control_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.video_progress_bar = ctk.CTkProgressBar(self.media_control_frame, orientation="horizontal",
                                                     mode="determinate")
        self.video_progress_bar.set(0)
        self.video_progress_bar.grid(row=0, column=0, columnspan=5, padx=(15, 15), pady=12, sticky="ew")

        self.lbl_video_time = ctk.CTkLabel(self.media_control_frame, text="00:00 / 00:00",
                                           font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_video_time.grid(row=0, column=5, padx=(5, 25), pady=12, sticky="e")

        self.btn_media_backward = ctk.CTkButton(self.media_control_frame, text="-5с", width=80,
                                                state="disabled", command=lambda: self.seek_media_action(-5))
        self.btn_media_backward.grid(row=1, column=1, padx=5, pady=2)

        self.btn_media_pause = ctk.CTkButton(self.media_control_frame, text="Пауза", width=100,
                                             state="disabled", command=self.toggle_pause_action)
        self.btn_media_pause.grid(row=1, column=2, padx=5, pady=2)

        self.btn_media_replay = ctk.CTkButton(self.media_control_frame, text="Повтор", width=80,
                                              state="disabled", command=self.replay_media_action)
        self.btn_media_replay.grid(row=1, column=3, padx=5, pady=2)

        self.btn_media_forward = ctk.CTkButton(self.media_control_frame, text="+5с", width=80,
                                               state="disabled", command=lambda: self.seek_media_action(5))
        self.btn_media_forward.grid(row=1, column=4, padx=5, pady=2)

        # --- ЖИВОЙ ГРАФИК ---
        self.chart_frame = ctk.CTkFrame(self.main_frame, fg_color="#1a1a1a", corner_radius=8)
        self.chart_frame.grid(row=2, column=0, sticky="nsew", pady=(5, 0))
        self.chart_frame.grid_columnconfigure(0, weight=1)
        self.chart_frame.grid_rowconfigure(0, weight=1)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.fig = Figure(figsize=(6, 2.5), dpi=100, facecolor='#1a1a1a')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a1a')

        self.bars = self.ax.barh(self.emotion_labels, [0] * 7, color='#1f538d')
        self.ax.set_xlim(0, 100)
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')

        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        # ==========================================
        # 3. ВКЛАДКА "ИСТОРИЯ СЕССИЙ"
        # ==========================================
        self.history_widget = HistoryWidget(self.tab_view.tab("История сессий"), db_manager=self.db_manager)
        self.history_widget.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_view.configure(command=self._on_tab_changed)

    def _on_tab_changed(self):
        if self.tab_view.get() == "История сессий":
            self.history_widget.refresh_list()

    def set_export_buttons_state(self, enabled: bool):
        state_val = "normal" if enabled else "disabled"
        self.btn_snapshot.configure(state=state_val)
        self.btn_export_report.configure(state=state_val)

        if enabled and isinstance(self.video_source, str):
            self.btn_export_video.configure(state="normal")
        else:
            self.btn_export_video.configure(state="disabled")

    def set_media_controls_state(self, is_file: bool):
        """Динамически показывает или скрывает панель плеера."""
        if is_file:
            self.media_control_frame.grid(row=1, column=0, sticky="ew", pady=5)
            self.btn_media_backward.configure(state="normal")
            self.btn_media_pause.configure(state="normal", text="Пауза")
            self.btn_media_replay.configure(state="normal")
            self.btn_media_forward.configure(state="normal")
        else:
            self.media_control_frame.grid_forget()

    def update_chart(self, emotion_scores: dict, dominant_emotion: str = None):
        heights = []
        for emotion in self.emotion_labels:
            score = emotion_scores.get(emotion, 0)
            if score <= 1.0 and score > 0:
                score *= 100
            heights.append(score)

        for bar, label, h in zip(self.bars, self.emotion_labels, heights):
            bar.set_width(h)
            if dominant_emotion and label == dominant_emotion.lower():
                bar.set_color('#00adb5')
            else:
                bar.set_color('#4a4a4a')

        self.canvas.draw()

    def toggle_camera(self):
        if self.worker and self.worker._is_running:
            self.worker.stop()
            self.btn_camera_mode.configure(text="Режим: Камера", fg_color=["#3a7ebf", "#1f538d"])
            self.video_label.configure(text="Камера остановлена", image="")
            self.emotion_status_label.configure(text="Эмоция: ---")
            self.update_chart({emotion: 0.0 for emotion in self.emotion_labels}, None)
            self.last_frame = None
            self.last_emotion_data = {}
            self.set_export_buttons_state(False)
            self.set_media_controls_state(False)
        else:
            self.smoother.clear()
            self.video_history_data.clear()
            self.video_source = int(self.combo_camera.get().split()[-1])
            self.worker = VideoWorker(source=self.video_source, engine=self.engine, smoother=self.smoother,
                                      frame_skip=self.current_frame_skip)
            self.worker.start()
            self.btn_camera_mode.configure(text="Остановить камеру", fg_color="#912424")
            self.set_media_controls_state(False)
            self.update_frame_loop()

    def toggle_pause_action(self):
        if self.worker and self.worker._is_running:
            if self.worker.is_paused():
                self.worker.set_paused(False)
                self.btn_media_pause.configure(text="Пауза", fg_color=["#3a7ebf", "#1f538d"])
            else:
                self.worker.set_paused(True)
                self.btn_media_pause.configure(text="Старт", fg_color="#27ae60")

    def replay_media_action(self):
        if self.worker and self.worker._is_running:
            self.worker.rewind_to_start()

            self.smoother.clear()
            self.update_chart({emotion: 0.0 for emotion in self.emotion_labels}, None)

            self.btn_media_pause.configure(text="Пауза", fg_color=["#3a7ebf", "#1f538d"])

            if hasattr(self, 'video_history_data'):
                self.video_history_data.clear()

    def seek_media_action(self, seconds: int):
        if self.worker and self.worker._is_running:
            self.worker.seek_relative(seconds)

    def update_frame_loop(self):
        if not self.worker or not self.worker._is_running:
            return

        if self.worker.is_paused():
            self.after(15, self.update_frame_loop)
            return

        if not self.worker.result_queue.empty():
            frame, smoothed_data, meta = self.worker.result_queue.get()

            # --- Обновление прогресс-бара и таймера ---
            if meta and meta["total_frames"] > 0:
                self.current_video_frame_idx = meta["current_frame"]
                total_frames = meta["total_frames"]
                fps = meta["fps"]

                current_sec = int(self.current_video_frame_idx / fps)
                total_sec = int(total_frames / fps)

                current_str = f"{current_sec // 60:02d}:{current_sec % 60:02d}"
                total_str = f"{total_sec // 60:02d}:{total_sec % 60:02d}"

                self.lbl_video_time.configure(text=f"{current_str} / {total_str}")

                # Передаем долю от 0.0 до 1.0 в ProgressBar
                progress_pct = self.current_video_frame_idx / total_frames
                self.video_progress_bar.set(progress_pct)

            dominant_emotion = smoothed_data.get('dominant_emotion', 'Unknown')
            region = smoothed_data.get('region')
            all_emotions = smoothed_data.get('emotion', {})

            is_emotion_valid = dominant_emotion and dominant_emotion != "Unknown"

            if region and is_emotion_valid:
                self.emotion_status_label.configure(text=f"Эмоция: {dominant_emotion.upper()}")
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                self.update_chart(all_emotions, dominant_emotion)

                # --- Шаг 2: Логирование в историю (для видео и камеры) ---
                if hasattr(self, 'video_history_data'):
                    # Считаем текущую секунду на основе индекса кадра и FPS (дефолт 30 если нет мета)
                    fps_val = meta["fps"] if (meta and "fps" in meta) else 30.0
                    timestamp_sec = round(self.current_video_frame_idx / fps_val, 2)

                    frame_entry = {
                        "frame_idx": self.current_video_frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "dominant_emotion": dominant_emotion,
                        # Фикс: Явно приводим v к float перед округлением, чтобы избавиться от float32 из numpy
                        "emotion_distribution": {k: round(float(v), 2) for k, v in all_emotions.items()}
                    }
                    self.video_history_data.append(frame_entry)
            else:
                self.emotion_status_label.configure(text="Лицо не найдено")
                self.update_chart({emotion: 0.0 for emotion in self.emotion_labels}, None)

            self.last_frame = frame.copy()
            self.last_emotion_data = smoothed_data
            self.set_export_buttons_state(True)

            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_img)

            max_w = max(self.main_frame.winfo_width(), 640)
            max_h = int(max(self.main_frame.winfo_height(), 480) * 0.7)

            ratio_w = max_w / img.width
            ratio_h = max_h / img.height
            scaling_factor = min(ratio_w, ratio_h)

            new_w = int(img.width * scaling_factor)
            new_h = int(img.height * scaling_factor)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))

            if self.worker and self.worker._is_running:
                self.video_label.configure(image=ctk_img, text="")

        if self.worker and self.worker._is_running:
            self.after(15, self.update_frame_loop)

    def change_camera_source(self, choice: str):
        try:
            new_source = int(choice.split()[-1])
            if self.video_source == new_source:
                return

            self.video_source = new_source
            print(f"[INFO] Источник камеры изменен на индекс: {self.video_source}")

            if self.worker and self.worker._is_running and isinstance(self.video_source, int):
                self.toggle_camera()
                self.toggle_camera()
        except Exception as e:
            print(f"[ERROR] Не удалось распарсить индекс камеры: {e}")

    def change_frameskip_value(self, value: float):
        self.current_frame_skip = int(value)
        self.lbl_frameskip.configure(text=f"Пропуск кадров: {self.current_frame_skip}")
        if self.worker and self.worker._is_running:
            self.worker.frame_skip = self.current_frame_skip

    def load_image_action(self):
        if self.worker and self.worker._is_running:
            self.worker.stop()
            self.btn_camera_mode.configure(text="Режим: Камера", fg_color=["#3a7ebf", "#1f538d"])

        self.smoother.clear()
        self.set_media_controls_state(False)

        file_path = filedialog.askopenfilename(
            title="Выберите фотографию для анализа",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.webp")]
        )

        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            return

        raw_results = self.engine.analyze(frame)

        if raw_results and len(raw_results) > 0 and 'dominant_emotion' in raw_results[0]:
            result = raw_results[0]
            dominant_emotion = result.get('dominant_emotion', 'Unknown')
            region = result.get('region', {})
            all_emotions = result.get('emotion', {})

            self.emotion_status_label.configure(text=f"Эмоция: {dominant_emotion.upper()}")
            self.update_chart(all_emotions, dominant_emotion)

            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            self.last_emotion_data = {"dominant_emotion": dominant_emotion, "emotion": all_emotions}
        else:
            self.emotion_status_label.configure(text="Лицо не найдено")
            self.update_chart({emotion: 0.0 for emotion in self.emotion_labels}, None)
            self.last_emotion_data = {"dominant_emotion": "Unknown", "emotion": {}}

        self.last_frame = frame.copy()
        self.set_export_buttons_state(True)

        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_img)

        max_w = max(self.main_frame.winfo_width(), 640)
        max_h = int(max(self.main_frame.winfo_height(), 480) * 0.7)

        ratio_w = max_w / img.width
        ratio_h = max_h / img.height
        scaling_factor = min(ratio_w, ratio_h)

        new_w = int(img.width * scaling_factor)
        new_h = int(img.height * scaling_factor)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
        self.video_label.configure(image=ctk_img, text="")

    def load_video_action(self):
        if self.worker and self.worker._is_running:
            self.worker.stop()
            self.btn_camera_mode.configure(text="Режим: Камера", fg_color=["#3a7ebf", "#1f538d"])

        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл для анализа",
            filetypes=[("Видео файлы", "*.mp4 *.avi *.mkv *.mov")]
        )

        if not file_path:
            return

        print(f"[INFO] Запуск анализа видеофайла: {file_path}")
        self.video_source = file_path
        self.smoother.clear()

        self.video_history_data.clear()
        self.worker = VideoWorker(source=file_path, engine=self.engine, smoother=self.smoother,
                                  frame_skip=self.current_frame_skip)
        self.worker.start()

        self.update_chart({emotion: 0.0 for emotion in self.emotion_labels}, None)
        self.last_frame = None
        self.last_emotion_data = {}
        self.set_export_buttons_state(False)
        self.set_media_controls_state(True)
        self.update_frame_loop()

    def save_snapshot_action(self):
        if self.last_frame is None:
            return
        try:
            path = self.exporter.save_snapshot(self.last_frame)
            print(f"[SUCCESS] Снимок сохранен: {path}")
            self.btn_snapshot.configure(text="Сохранено!", fg_color="#27ae60")
            self.after(1000, lambda: self.btn_snapshot.configure(text="Сделать снимок", fg_color="#2c3e50"))
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить снимок: {e}")

    def save_report_action(self):
        # Разрешаем экспорт, если есть ХОТЯ БЫ данные последнего кадра ИЛИ накопленная история
        has_history = hasattr(self, 'video_history_data') and self.video_history_data
        if not self.last_emotion_data and not has_history:
            return

        try:
            import datetime
            now_iso = datetime.datetime.now().isoformat()

            # Сценарий 1: Есть накопленная история (видео или камера)
            if has_history:
                total_frames = len(self.video_history_data)
                emotions_count = {}
                for entry in self.video_history_data:
                    dom = entry["dominant_emotion"]
                    emotions_count[dom] = emotions_count.get(dom, 0) + 1

                most_dominant = max(emotions_count, key=emotions_count.get) if emotions_count else "Unknown"

                report_data = {
                    "project_name": "Emotion Analytics",
                    "source": str(self.video_source),
                    "export_timestamp": now_iso,
                    "summary": {
                        "total_frames_with_faces": total_frames,
                        "most_dominant_emotion_overall": most_dominant,
                        "emotions_occurrence_count": emotions_count
                    },
                    "timeline": self.video_history_data
                }

            # Сценарий 2: Истории нет, это одиночное фото
            else:
                # Берем доминирующую эмоцию и распределение из последнего кадра
                dominant_emotion = self.last_emotion_data.get("dominant_emotion", "Unknown")
                raw_emotions = self.last_emotion_data.get("emotion", {})

                # Приводим к float (зачищаем numpy.float32, если они проскочили)
                clean_distribution = {k: round(float(v), 2) for k, v in raw_emotions.items()}

                # Формируем аналогичную структуру, чтобы БД не падала
                report_data = {
                    "project_name": "Single Photo Analytics",
                    "source": "Image File",
                    "export_timestamp": now_iso,
                    "summary": {
                        "total_frames_with_faces": 1,
                        "most_dominant_emotion_overall": dominant_emotion,
                        "emotions_occurrence_count": {dominant_emotion: 1} if dominant_emotion != "Unknown" else {}
                    },
                    "timeline": [
                        {
                            "frame_idx": 0,
                            "timestamp_sec": 0.0,
                            "dominant_emotion": dominant_emotion,
                            "emotion_distribution": clean_distribution
                        }
                    ]
                }

            # --- Шаг 1: Сохранение в JSON файл ---
            path = self.exporter.save_report(report_data)
            print(f"[SUCCESS] Отчет сохранен в файл: {path}")

            # --- Шаг 2: Дублирование в Базу Данных ---
            session_id = self.db_manager.save_session(report_data)
            print(f"[SUCCESS] Сессия успешно записана в БД под ID: {session_id}")

            # Изменяем состояние кнопки для индикации успеха
            self.btn_export_report.configure(text="Успешно!", fg_color="#27ae60")
            self.after(1000,
                       lambda: self.btn_export_report.configure(text="Экспорт аналитики (JSON)", fg_color="#2c3e50"))

        except Exception as e:
            print(f"[ERROR] Не удалось сохранить отчет: {e}")

    def save_video_action(self):
        """Запуск процесса экспорта с кастомным диалоговым окном."""
        if not isinstance(self.video_source, str):
            return

        # Вызываем модальное окно кастомного стиля
        dialog = CustomExportDialog(self)
        self.wait_window(dialog)

        choice = dialog.result
        if choice is None:
            return

        limit_frames = None
        if choice == "part":
            limit_frames = self.current_video_frame_idx
            if limit_frames <= 0:
                messagebox.showwarning("Внимание", "Плеер находится в самом начале видео.")
                return

        output_path = filedialog.asksaveasfilename(
            title="Сохранить обработанное видео",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )

        if not output_path:
            return

        if self.worker and self.worker._is_running:
            self.worker.stop()

        self.btn_export_video.configure(state="disabled", text="Экспорт...")
        self.btn_load_video.configure(state="disabled")
        self.btn_camera_mode.configure(state="disabled")

        self.progress_bar.grid(row=14, column=0, padx=20, pady=5, sticky="ew")
        self.progress_bar.set(0)

        self.exporter_worker = VideoExportWorker(
            source_path=self.video_source,
            output_path=output_path,
            engine=self.engine,
            limit_frames=limit_frames
        )
        self.exporter_worker.start(
            on_progress=self._on_export_progress,
            on_finished=self._on_export_finished
        )

    def _on_export_progress(self, value: float):
        self.after(0, lambda: self.progress_bar.set(value))

    def _on_export_finished(self):
        """Вызывается при успешном завершении рендеринга."""

        def ui_finish():
            self.progress_bar.grid_forget()

            self.btn_load_video.configure(state="normal")
            self.btn_camera_mode.configure(state="normal")
            self.btn_export_video.configure(text="Сохранено!", fg_color="#27ae60")

            # --- ВОССТАНОВЛЕНИЕ РАБОТЫ ПЛЕЕРА ---
            # Инициализируем поток заново, так как старый был полностью остановлен
            self.worker = VideoWorker(source=self.video_source, engine=self.engine, smoother=self.smoother,
                                      frame_skip=self.current_frame_skip)
            self.worker.start()

            # Ставим на паузу и перематываем на тот кадр, где остановились
            self.worker.set_paused(True)
            if self.current_video_frame_idx > 0:
                # Переводим индекс кадра в секунды для корректного seek
                fps = 30.0  # Дефолтное значение на случай, если мета еще не пришла
                if hasattr(self, 'exporter_worker') and self.exporter_worker:
                    # Попробуем взять точный FPS из исходного видео через экспортер
                    cap = cv2.VideoCapture(self.video_source)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()

                target_seconds = self.current_video_frame_idx / fps
                self.worker.seek_relative(int(target_seconds))

            # Активируем кнопки интерфейса
            self.set_media_controls_state(True)
            self.btn_media_pause.configure(text="Старт", fg_color="#27ae60")

            # Запускаем цикл обновления кадров заново
            self.update_frame_loop()

            self.after(1500, lambda: self.btn_export_video.configure(text="Сохранить видео", fg_color="#2c3e50"))

        self.after(0, ui_finish)

    def destroy(self):
        if self.worker:
            self.worker.stop()
        super().destroy()
