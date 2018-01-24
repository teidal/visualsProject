import numpy as np
import cv2


def take_photo():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    photo = ""
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF # wait for user input

        if k == ord('t'): # if t, take picture
            print("Taking image...")
            photo = "pictures\photo.png"
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(photo, frame)
            print("Done!")
        elif k == ord('q'): # if q quit
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return photo
