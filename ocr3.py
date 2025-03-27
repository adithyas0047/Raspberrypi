import cv2
import pytesseract
from picamera2 import Picamera2
from gtts import gTTS
from playsound import playsound
import numpy as np
import os
from vosk import Model, KaldiRecognizer
import pyaudio

# Load Vosk Model
model_path = "/home/dell/vosk-model-small-en-us"  # Change this path if needed
if not os.path.exists(model_path):
    print("Vosk model not found! Download from https://alphacephei.com/vosk/models")
    exit(1)

model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

# Initialize microphone input
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

# Initialize Picamera2 but don't start it yet
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))

def text_to_speech(text):
    """Convert recognized text to speech and play it."""
    if text.strip():
        tts = gTTS(text, lang='en')
        audio_file = '/tmp/ocr_result.mp3'
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
    else:
        print("No recognizable text found.")

print("Say 'start' to open the camera, or 'stop' to capture and process...")

camera_active = False

try:
    while True:
        print("Listening for command...")
        data = stream.read(4096, exception_on_overflow=False)

        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            command = result.split('"')[3].strip().lower()  # Extract recognized text
            print(f"Recognized: {command}")

            if 'start' in command and not camera_active:
                print("Starting camera...")
                picam2.start()
                camera_active = True

            elif 'stop' in command and camera_active:
                print("Capturing image and processing...")
                
                # Capture image
                frame = picam2.capture_array()
                picam2.stop()
                camera_active = False

                # Convert to grayscale for OCR
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Perform OCR
                text = pytesseract.image_to_string(gray)
                print("OCR Result:\n", text)

                # Read out the recognized text
                text_to_speech(text)

                # Show captured image with OCR
                cv2.imshow("Captured Image", gray)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    stream.stop_stream()
    stream.close()
    mic.terminate()