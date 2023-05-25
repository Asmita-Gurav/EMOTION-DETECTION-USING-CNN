import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
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
    global cap, canvas, timer, emoji_label, emoji_images

    # Define the emoji images
    happy_emoji = Image.open("emoji/happy.png")
    sad_emoji = Image.open("emoji/sad.png")
    neutral_emoji = Image.open("emoji/neutral.png")
    angry_emoji = Image.open("emoji/angry.png")
    surprised_emoji = Image.open("emoji/surprised.png")
    disgusted_emoji = Image.open("emoji/disgusted.png")
    fearful_emoji = Image.open("emoji/fearful.png")
    emoji_images = [angry_emoji, disgusted_emoji, fearful_emoji, happy_emoji, neutral_emoji, sad_emoji, surprised_emoji]

    # Open the video stream
    cap = cv2.VideoCapture(0)

    # Update the video stream on the canvas and emoji
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

                # Update the emoji image based on the predicted emotion
                emoji_image = emoji_images[maxindex]
                emoji_label.configure(image=emoji_image)
                emoji_label.image = emoji_image

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = imgtk
            canvas.configure(image=imgtk)
        timer = canvas.after(10, update)

    # Create a quit button to stop the video capture
    def quit_capture():
        cap.release()
        root.destroy()

    # Create the root window
    root = tk.Tk()

    # Create the canvas to display the video stream
    canvas = tk.Canvas(root, width=600, height=400)
    canvas.pack()

    # Create the emoji label to display the predicted emotion
    emoji_image = Image.open("emoji/neutral.png")
    emoji_image = emoji_image.resize((100, 100))
    emoji_photo = ImageTk.PhotoImage(emoji_image)
    emoji_label = tk.Label(root, image=emoji_photo)
    emoji_label.pack()

    # Start capturing the video stream
    start_capture()

    # Run the root window
    root.mainloop()

