import sys, os
from unittest.mock import MagicMock

class MockWebview:
    def create_window(self, *args, **kwargs):
        class W:
            def evaluate_js(self, code):
                print(f"[JS EVAL] {code}")
        return W()
    def start(self, *args, **kwargs):
        pass

sys.modules['webview'] = MockWebview()
sys.modules['pyaudio'] = MagicMock()
sys.modules['openwakeword'] = MagicMock()
sys.modules['openwakeword.model'] = MagicMock()
sys.modules['openwakeword.utils'] = MagicMock()
sys.modules['faster_whisper'] = MagicMock()

import jarvis

def test():
    jarvis.bootstrap_memory()
    window = MockWebview().create_window()
    jarvis.WINDOW = window
    print("Testing process_command('Hey Jarvis')")
    try:
        jarvis.process_command("Hey Jarvis", window)
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

test()
