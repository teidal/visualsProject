import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

NORM_FONT = ("Verdana", 10)
root = tk.Tk()
root.withdraw()


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("Camera")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    button1 = ttk.Button(popup, text="OK", command=popup.destroy())
    button1.pack()
    popup.mainloop()


def take_photo():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    photo = ""
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Camera', frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('t'):
            print("Taking image...")
            photo = "pictures\photo.png"
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(photo, frame)
            print("Done!")
            messagebox.showinfo("Camera", "Photo successfully taken")
        elif k == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return photo


if __name__ == "__main__":
    take_photo()
