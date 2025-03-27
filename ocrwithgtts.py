import cv2
import pytesseract
from picamera2 import Picamera2
from gtts import gTTS
from playsound import playsound
import numpy as np
import os

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

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

print("Press 's' to capture and perform OCR, or 'q' to quit...")

try:
    while True:
        # Capture frame from the camera
        frame = picam2.capture_array()

        # Display the frame
        cv2.imshow("Frame", frame)

        # Wait for key event
        key = cv2.waitKey(1) & 0xFF

        # Capture and perform OCR when 's' key is pressed
        if key == ord('s'):
            # Convert the image to grayscale for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Perform OCR
            text = pytesseract.image_to_string(gray)
            print("OCR Result:\n", text)

            # Read out the recognized text
            text_to_speech(text)

            # Show captured image with OCR
            cv2.imshow("Captured Frame for OCR", gray)

        # Exit the loop when 'q' key is pressed
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Release resources
    cv2.destroyAllWindows()
    picam2.stop()