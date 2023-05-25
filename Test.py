import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import model_from_json

# Define emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the pre-trained model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")


# Define function to capture video
def start_capture():
    global cap, canvas, timer

    # Open the video stream
    cap = cv2.VideoCapture(0)

    # Update the video stream on the canvas
    def update():
        _, frame = cap.read()
        if _:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = imgtk
            canvas.configure(image=imgtk)
        timer = canvas.after(10, update)

    update()


# Define function to stop capturing video
def stop_capture():
    global cap, canvas, timer

    # Stop the video stream
    cap.release()

    # Stop updating the video stream on the canvas
    if timer:
        canvas.after_cancel(timer)
        timer = None


# Create the GUI
root = tk.Tk()
root.title("Emotion Detection")

# Create a canvas for the video stream
canvas = tk.Label(root, bd=0)
canvas.pack()

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Create the buttons2
start_button = tk.Button(button_frame, text="Start", command=start_capture)
start_button.pack(side=tk.LEFT, padx=10)
stop_button = tk.Button(button_frame, text="Stop", command=stop_capture)
stop_button.pack(side=tk.LEFT, padx=10)
close_button = tk.Button(button_frame, text="Close", command=root.destroy)
close_button.pack(side=tk.LEFT, padx=10)

# Start the GUI
root.mainloop()
