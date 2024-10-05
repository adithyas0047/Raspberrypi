from gtts import gTTS
from playsound import playsound
import face_recognition
import imutils
import pickle
import time
import cv2
import subprocess
import numpy as np
import os

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
encodingsP = "encodings.pickle"

# Load the known faces and embeddings
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Function to capture a frame using libcamera
def capture_frame():
    subprocess.run(["libcamera-still", "-o", "/tmp/frame.jpg", "-t", "1"], check=True)
    frame = cv2.imread("/tmp/frame.jpg")
    return frame

# Function to convert text to audio and play it
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    audio_file = '/tmp/announcement.mp3'
    tts.save(audio_file)
    playsound(audio_file)
    # Clean up the file
    os.remove(audio_file)

frame = capture_frame()
if frame is None:
    print("[ERROR] No frame captured from the camera. Exiting...")
    exit(1)

fps = cv2.getTickFrequency()

while True:
    frame = capture_frame()
    if frame is None:
        print("[ERROR] No frame captured from camera. Exiting...")
        break

    frame = imutils.resize(frame, width=500)
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            if currentname != name:
                currentname = name
                announcement = f"{currentname} is approaching you"
                print(announcement)
                text_to_speech(announcement)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps = cv2.getTickCount() - fps

print("[INFO] elapsed time: {:.2f}".format(fps / cv2.getTickFrequency()))
print("[INFO] approx. FPS: {:.2f}".format(cv2.getTickFrequency() / fps))

cv2.destroyAllWindows()
