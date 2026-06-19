import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from database.manager import DatabaseManager


class HistoryWidget(ctk.CTkFrame):
    def __init__(self, parent, db_manager: DatabaseManager):
        super().__init__(parent, fg_color="transparent")
        self.db_manager = db_manager
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.grid_columnconfigure(0, weight=4)  # Левая часть со списком
        self.grid_columnconfigure(1, weight=6)  # Правая часть с графиком
        self.grid_rowconfigure(0, weight=1)

        # === ЛЕВАЯ СТОРОНА: Список сессий ===
        self.left_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=8)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)

        self.title_lbl = ctk.CTkLabel(self.left_frame, text="Архив сохраненных сессий",
                                      font=ctk.CTkFont(size=15, weight="bold"))
        self.title_lbl.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        # Прокручиваемый контейнер для списка строк
        self.scroll_container = ctk.CTkScrollableFrame(self.left_frame, fg_color="transparent")
        self.scroll_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.scroll_container.grid_columnconfigure(0, weight=1)

        # === ПРАВАЯ СТОРОНА: График сессии ===
        self.right_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=8)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

        self.chart_lbl = ctk.CTkLabel(self.right_frame, text="Итоговая аналитика сессии",
                                      font=ctk.CTkFont(size=15, weight="bold"))
        self.chart_lbl.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        # Инициализируем Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#1a1a1a')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a1a')
        self.bars = self.ax.barh(self.emotion_labels, [0] * 7, color='#4a4a4a')
        self.ax.set_xlim(0, 100)
        self.ax.tick_params(colors='white', labelsize=10)

        for spine in ['top', 'right']: self.ax.spines[spine].set_visible(False)
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.fig.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.15)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        self.active_session_id = None
        self.session_buttons = {}

    def refresh_list(self):
        """Загружает сессии из БД и обновляет список в интерфейсе."""
        # Очищаем контейнер от старых виджетов
        for widget in self.scroll_container.winfo_children():
            try:
                widget.destroy()
            except Exception:
                pass  # Защита от разрушения недоинициализированных элементов

        self.session_buttons.clear()

        try:
            sessions = self.db_manager.get_all_sessions()
            if not sessions:
                no_data_lbl = ctk.CTkLabel(self.scroll_container, text="История пуста.\nЭкспортируйте отчет в плеере.",
                                           text_color="gray")
                no_data_lbl.pack(pady=40)
                return

            for idx, s in enumerate(sessions):
                # Форматируем красивую строку для кнопки
                date_raw = s["timestamp"].split("T")[0]
                time_raw = s["timestamp"].split("T")[1][:5]
                src_short = s["source"].split("/")[-1].split("\\")[-1]  # Очищаем путь до имени файла
                if len(src_short) > 20: src_short = src_short[:17] + "..."

                btn_text = f" ID {s['id']} | {date_raw} {time_raw}\n Из: {src_short}\n Доминирует: {s['dominant_emotion'].upper()} ({s['total_frames']} кадр.)"

                # Фрейм-строка для размещения кнопки удаления и выбора
                item_frame = ctk.CTkFrame(self.scroll_container, fg_color="transparent")
                item_frame.pack(fill="x", pady=4)
                item_frame.grid_columnconfigure(0, weight=1)

                # ФИКС: Убран параметр justify, который вешал customtkinter
                btn = ctk.CTkButton(item_frame, text=btn_text, anchor="w",
                                    fg_color="#2b2b2b", hover_color="#3a3a3a", font=ctk.CTkFont(size=11),
                                    command=lambda s_id=s["id"]: self.select_session(s_id))
                btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))

                del_btn = ctk.CTkButton(item_frame, text="✕", width=30, fg_color="#551a1a", hover_color="#822424",
                                        command=lambda s_id=s["id"]: self.delete_session_action(s_id))
                del_btn.grid(row=0, column=1, sticky="ns")

                self.session_buttons[s["id"]] = btn

            if self.active_session_id and self.active_session_id in self.session_buttons:
                self.session_buttons[self.active_session_id].configure(fg_color="#1f538d")
        except Exception as e:
            print(f"[ERROR] Ошибка обновления списка истории: {e}")

    def select_session(self, session_id: int):
        """Вызывается при клике на сессию. Подсвечивает кнопку и строит график."""
        if self.active_session_id in self.session_buttons:
            self.session_buttons[self.active_session_id].configure(fg_color="#2b2b2b")

        self.active_session_id = session_id
        if session_id in self.session_buttons:
            self.session_buttons[session_id].configure(fg_color="#1f538d")

        # Загружаем полные данные
        session = self.db_manager.get_session_details(session_id)
        if not session:
            return

        summary = session.get("emotions_summary", {})
        total = sum(summary.values()) if sum(summary.values()) > 0 else 1

        heights = []
        for emotion in self.emotion_labels:
            count = summary.get(emotion, 0)
            percentage = (count / total) * 100
            heights.append(percentage)

        dom_emotion = session.get("dominant_emotion", "").lower()
        self.chart_lbl.configure(
            text=f"Итоговая аналитика сессии ID {session_id} (Всего кадров: {session['total_frames']})")

        # Перерисовываем график
        for bar, label, h in zip(self.bars, self.emotion_labels, heights):
            bar.set_width(h)
            bar.set_color('#00adb5' if label == dom_emotion else '#4a4a4a')

        self.canvas.draw()

    def delete_session_action(self, session_id: int):
        """Удаляет запись и сбрасывает график, если удалили активную."""
        self.db_manager.delete_session(session_id)
        if self.active_session_id == session_id:
            self.active_session_id = None
            for bar in self.bars: bar.set_width(0)
            self.canvas.draw()
            self.chart_lbl.configure(text="Итоговая аналитика сессии")
        self.refresh_list()