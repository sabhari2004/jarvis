import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

print("Locating JARVIS Wake Word Model...")

# 1. Ask openWakeWord where it installed its internal pre-trained models
model_paths = openwakeword.get_pretrained_model_paths()

# 2. Search that list specifically for the Jarvis model (to save RAM)
jarvis_path = next((m for m in model_paths if "hey_jarvis" in m.lower()), None)

if not jarvis_path:
    print("Error: Could not find the included hey_jarvis model!")
    exit(1)

print(f"Loading model from: {jarvis_path}")

# 3. Load the model using the exact absolute path
owwModel = Model(wakeword_model_paths=[jarvis_path])

# Setup the microphone
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n🟢 Sleeper Agent Active. Listening for 'Hey Jarvis'...")

while True:
    # Capture audio chunks
    audio_data = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    
    # Analyze audio for the wake word
    prediction = owwModel.predict(audio_data)
    
    # Check if the AI is confident we said "Hey Jarvis"
    for word, score in prediction.items():
        if score > 0.5:
            print(f"\n*** WAKE WORD DETECTED! (Confidence: {score:.2f}) Triggering UI... ***")
            
            # Flush the audio buffer so it doesn't trigger twice in a row
            mic_stream.read(CHUNK * 10, exception_on_overflow=False)