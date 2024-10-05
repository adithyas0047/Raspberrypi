import cv2
from picamera2 import Picamera2
import time
import os

name = 'himali'  # replace with your name
save_dir = f'dataset/{name}/'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

cam = Picamera2()
config = cam.create_still_configuration(main={"size": (512, 304)})
cam.configure(config)
cam.start()

img_counter = 0

while True:
    frame = cam.capture_array()
    cv2.imshow("Press Space to take a photo", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:  # SPACE pressed
        img_name = f"{save_dir}/image_{img_counter}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

cam.stop()
cv2.destroyAllWindows()