import cv2
from card import *

# Initialize main camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera can't open")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame, exiting...")
        break

    # opens window with camera feed
    processed_frame = preprocess_image(frame)
    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Preprocess', processed_frame)

    # "q" quits the window
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()