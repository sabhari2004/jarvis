import sys
import jarvis
from unittest.mock import MagicMock

window = MagicMock()
jarvis.WINDOW = window
jarvis.process_command("Hey Jarvis", window)
print("Finished!")
