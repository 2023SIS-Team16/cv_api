# from hand_segment import HandSegmenter

import socket, os
import cv2 as cv
import numpy as np

# UDP_IP = "127.0.0.1"
# UDP_PORT = 54174

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# sock.bind((UDP_IP, UDP_PORT))

# print("UDP target IP:", UDP_IP)

# while True:
#     print("Waiting for data...")
#     data, addr = sock.recvfrom(10000000) # buffer size is 1024 bytes
#     for x in data:
#         print(x)

# sock.close()
# os.system("pause")

from model.data.inference_testing.landmarks import LandmarkProcessor
import tensorflow as tf

processor = LandmarkProcessor(
    pose_landmarker="/Users/jon/development/university/sis/models/pose_landmarker_full.task",
    hand_landmarker="/Users/jon/development/university/sis/models/hand_landmarker.task",
    face_landmarker="/Users/jon/development/university/sis/models/face_landmarker.task"
)

model_path = "/Users/jon/development/university/sis/models/cnn/model_2.tflite"

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

interpreter = tf.lite.Interpreter(model_path=model_path)

input_details = interpreter.get_input_details()
print(input_details)

found_signatures = list(interpreter.get_signature_list().keys())
print(found_signatures)
print(interpreter.get_signature_list())

RUNNER_SIGNATURE = "serving_default"
REQ_OUTPUT = "outputs"

if RUNNER_SIGNATURE not in found_signatures:
    raise ValueError(f"Signature {RUNNER_SIGNATURE} not found in model. Found {found_signatures}")

prediction_fn = interpreter.get_signature_runner(RUNNER_SIGNATURE)
print(prediction_fn)

phrase = []

cam = cv.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cam.read()
    if frame_count == 10:
        frame_count = 0
    else:
        frame_count += 1
        continue

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    pose_indices = []#[12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21]
    hand_indices = list(range(0, 21))

    # cv.imshow('frame', frame)
    pose_landmarks, hand_landmarks, _, _ = processor.get_landmarks(frame)
    if hand_landmarks == [] or len(hand_landmarks) < 1 or pose_landmarks == [] or len(pose_landmarks) < 1:
        print(f"No landmarks: {len(hand_landmarks)} ... {len(pose_landmarks)}")
        # cv.destroyAllWindows()
        continue # Skip this image if there are no hand landmarks


    landmarks = []
    landmarks.extend(hand_landmarks[0].landmark[i].x for i in hand_indices)
    landmarks.extend(pose_landmarks[i].y for i in pose_indices)

    landmarks.extend(hand_landmarks[0].landmark[i].y for i in hand_indices)
    landmarks.extend(pose_landmarks[i].y for i in pose_indices)

    landmarks.extend(hand_landmarks[0].landmark[i].z for i in hand_indices)
    landmarks.extend(pose_landmarks[i].z for i in pose_indices)
    landmarks = np.array([landmarks])

    prediction_data = tf.cast(landmarks, tf.float32)
    prediction_data = tf.reshape(prediction_data, (prediction_data.shape[0], prediction_data.shape[1], 1))

    prediction = prediction_fn(conv1d_8_input=prediction_data)['dense_3']

    cats = np.zeros((1, 26))
    cats[np.arange(1), np.argmax(prediction)] = 1
    print(cats)

    letter = ALPHABET[np.argmax(prediction)]
    print(letter)

    if len(phrase) == 0 or phrase[-1] != letter:
        phrase.append(letter)
        print(phrase)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

