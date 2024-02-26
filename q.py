import time
import sys
import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import threading
import speech_recognition as sr

in_progress = []


def get_name(r, known_face_encodings, known_face_names, face_encoding):
    try:
        with sr.Microphone() as source2:
            time.sleep(0.5)
            r.adjust_for_ambient_noise(source2, duration=2)

            print("start")
            audio2 = r.listen(source2)

            text = r.recognize_google(audio2)
            text = text.lower()

            known_face_encodings.append(face_encoding)
            known_face_names.append(text)
            in_progress.remove(face_encoding)

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        in_progress.remove(face_encoding)

    except sr.UnknownValueError:
        print("unknown error occurred")
        in_progress.remove(face_encoding)


r = sr.Recognizer()
video_capture = cv2.VideoCapture(0)

obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding, (left, top, right, bottom) in zip(
            face_encodings, face_locations
        ):
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)

            else:
                face_names.append("detecting... ")
                if in_progress != []:
                    continue
                in_progress.append(face_encoding)
                name_thread = threading.Thread(
                    target=get_name,
                    args=(
                        r,
                        known_face_encodings,
                        known_face_names,
                        face_encoding,
                    ),
                )
                name_thread.start()

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
