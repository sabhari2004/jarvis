import jarvis

class DummyWindow:
    def evaluate_js(self, js):
        print(f"JS: {js}")

window = DummyWindow()
jarvis.WINDOW = window
jarvis.process_command("Hey Jarvis", window)
