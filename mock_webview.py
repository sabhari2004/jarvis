import sys
from unittest.mock import MagicMock
sys.modules['webview'] = MagicMock()
sys.modules['openwakeword'] = MagicMock()
sys.modules['openwakeword.model'] = MagicMock()

import jarvis

class DummyWindow:
    def evaluate_js(self, js):
        print("JS:", js)

window = DummyWindow()
jarvis.WINDOW = window
print("Testing handle_hey_jarvis")
jarvis.handle_hey_jarvis(window, "hey jarvis")
print("Done")
