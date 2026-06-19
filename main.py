from gui.app import EmotionApp


def main():
    # Создаем экземпляр нашего приложения CustomTkinter
    app = EmotionApp()

    # Запускаем главный цикл обработки событий Windows
    app.mainloop()


if __name__ == "__main__":
    main()