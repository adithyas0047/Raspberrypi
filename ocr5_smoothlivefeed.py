import cv2
import pytesseract
from picamera2 import Picamera2
from gtts import gTTS
from playsound import playsound
import numpy as np
import os
from vosk import Model, KaldiRecognizer
import pyaudio
import json

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

# Initialize Picamera2 with higher frame rate
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}, controls={"FrameRate": 30})
picam2.configure(camera_config)

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
            result_json = recognizer.Result()
            result = json.loads(result_json)  # Convert JSON result to dictionary
            command = result.get("text", "").strip().lower()  # Extract recognized text
            print(f"Recognized: {command}")

            if 'start' in command and not camera_active:
                print("Starting camera...")
                picam2.start()
                camera_active = True

                while camera_active:
                    request = picam2.capture_request()
                    frame = request.make_array("main")  # More optimized frame capture
                    request.release()

                    cv2.imshow("Live Camera Feed", frame)
                    key = cv2.waitKey(1)  # Reduces delay, improving frame rate

                    # Check if "stop" is spoken
                    data = stream.read(4096, exception_on_overflow=False)
                    if recognizer.AcceptWaveform(data):
                        result_json = recognizer.Result()
                        result = json.loads(result_json)
                        command = result.get("text", "").strip().lower()

                        if 'stop' in command:
                            print("Capturing image and processing...")

                            # Convert to grayscale for OCR
                            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                            # Perform OCR
                            text = pytesseract.image_to_string(gray)
                            print("OCR Result:\n", text)

                            # Read out the recognized text
                            text_to_speech(text)

                            # Show captured image for OCR feedback
                            cv2.imshow("Captured Image", gray)
                            cv2.waitKey(2000)
                            cv2.destroyAllWindows()

                            camera_active = False
                            picam2.stop()
                            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    stream.stop_stream()
    stream.close()
    mic.terminate()