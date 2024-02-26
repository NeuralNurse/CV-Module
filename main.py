import time
import sys
import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import threading
import speech_recognition as sr
from ultralytics import YOLO
import math
import pyttsx3
from summarizer import extract_information
from items import Item
import os


def speak_text(command):
	os.system("say" " " + command) 

def speak_thread(command):
	speak_thread = threading.Thread(
		target=speak_text, args=(command,)
		)
	speak_thread.start()

class Camera:
	def __init__(self):
		self.in_progress = []
		self.r = sr.Recognizer()
		self.video_capture = cv2.VideoCapture(0)
		self.w = 1280
		self.h = 720
		self.video_capture.set(3, self.w)
		self.video_capture.set(4, self.h)
		self.set_up_recognition()
		self.model = YOLO("../YOLO Weights/yolov8s.pt")
		with open('classes.txt') as f:
			self.classes = f.read().splitlines()
		self.name_queue = []
		self.speaking_queue = []
		self.item_already_said = []
		self.name_already_said = []
		tag = threading.Thread(target=self.speaker_thread, args=())
		tag.start()


	def set_up_recognition_backup(self):
		obama_image = face_recognition.load_image_file("obama.jpg")
		obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

		biden_image = face_recognition.load_image_file("biden.jpg")
		biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

		self.known_face_encodings = [obama_face_encoding, biden_face_encoding]
		self.known_face_names = ["Barack Obama", "Joe Biden"]
		self.process_this_frame = True

	def set_up_recognition(self):

		self.known_face_encodings = np.load('face_encodings.npy', allow_pickle=True).tolist()
		with open('face_names.txt', 'r') as f:
			self.known_face_names = f.read().splitlines()
		if len(self.known_face_encodings) != len(self.known_face_names) or self.known_face_names == []:
			self.set_up_recognition_backup()
		self.process_this_frame = True

	def save_data(self):
		np.save('face_encodings.npy', np.array(self.known_face_encodings, dtype=object), allow_pickle=True)
		with open('face_names.txt', 'w') as f:
		    for face in self.known_face_names:
		        f.write(f"{face}\n")
	def get_name(self, face_encoding):
		try:
			with sr.Microphone() as source2:
				time.sleep(0.5)
				self.r.adjust_for_ambient_noise(source2, duration=2)

				print("start")
				audio2 = self.r.listen(source2)

				text = self.r.recognize_google(audio2)
				text = text.lower()

				info = extract_information(text)
				if "unknown" in info.lower():
					self.in_progress.remove(face_encoding)
					print("error")
					return
				if ":" in info:
					name, relationship = info.split(":")
				elif "," in info:
					name, relationship = info.split(",")
				else:
					self.in_progress.remove(face_encoding)
					print("error")
					return
				text = f"{name}: {relationship}"

				self.known_face_encodings.append(face_encoding)
				self.known_face_names.append(text)
				self.in_progress.remove(face_encoding)
				print("end")

		except sr.RequestError as e:
			print("Could not request results; {0}".format(e))
			self.in_progress.remove(face_encoding)

		except sr.UnknownValueError:
			print("unknown error occurred")
			self.in_progress.remove(face_encoding)

	def speaker_thread(self):
		while True:
			for obj in self.name_queue:
				side, name = obj
				speak_thread(f"On your {side} is {name}")
				self.name_queue.remove([side, name])
				time.sleep(4)
			for obj in self.speaking_queue:
				speak_thread(f"On your {obj.side} is a {obj.item}")
				self.speaking_queue.remove(obj)


	def main_loop(self):
		while True:
			ret, frame = self.video_capture.read()
			if self.process_this_frame:
				results = self.model(frame, stream=True, verbose=False)

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
						self.known_face_encodings, face_encoding
					)
					face_distances = face_recognition.face_distance(
						self.known_face_encodings, face_encoding
					)
					best_match_index = np.argmin(face_distances)
					if matches[best_match_index]:
						name = self.known_face_names[best_match_index]
						face_names.append(name)

					else:
						face_names.append("Detecting... ")
						if self.in_progress != []:
							continue
						self.in_progress.append(face_encoding)
						name_thread = threading.Thread(
							target=self.get_name, args=(face_encoding,)
						)
						name_thread.start()

			self.process_this_frame = not self.process_this_frame

			for (top, right, bottom, left), name in zip(face_locations, face_names):
				top *= 4
				right *= 4
				bottom *= 4
				left *= 4
				midpoint = left + ((right - left) / 2)
				if (midpoint / self.w > 0.66):
					side = "left"
				elif (midpoint / self.w > 0.33):
					side = "middle"
				else:
					side = "right"
				if(name in self.name_already_said):
					pass
				else:
					self.name_already_said.append(name)
					self.name_queue.append([side, name])
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

				cv2.rectangle(
					frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
				)
				font = cv2.FONT_HERSHEY_DUPLEX
				cv2.putText(
					frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
				)
			for r in results:
				boxes = r.boxes
				for box in boxes:
					object_cls = box.cls[0]
					name = self.classes[int(object_cls)]
					if name == "person":
						continue
					x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
					midpoint = x1 + ((x2 - x1) / 2)
					if (midpoint / self.w > 0.66):
						side = "left"
					elif (midpoint / self.w > 0.33):
						side = "middle"
					else:
						side = "right"
					it = Item(name, side)
					if(it.hash in self.item_already_said):
						pass
					else:
						self.speaking_queue.append(it)
						self.item_already_said.append(it.hash)
					cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)

					confidence = math.ceil((box.conf[0] * 100)) / 100


					cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
			cv2.imshow("Video", frame)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

		self.save_data()
		self.video_capture.release()
		cv2.destroyAllWindows()

n = Camera()
n.main_loop()
